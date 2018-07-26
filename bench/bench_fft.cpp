#include <kfr/dft.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/sem.h>
#include <sys/shm.h>

//#include <omp.h>
#include <fftw3.h>

const int CHANNELS = 32;
const int REPEATS = 1000;
const int MAX = 10000000;
const int FFTW_FLAGS = FFTW_DESTROY_INPUT | FFTW_MEASURE;

// zrobić porównanie bardziej ujednolicone
// i zrobić porównanie z wersją MPI z własną komunikacją

// SPRAWDZAMY NA RAZIE POD KĄTEM EMPI (raczej-małe transformaty)
// funkcja którą benchmarkujemy powinna wyglądać tak:
// void compute(int C, int N, double* input, fftw_complex* output);
// i ustalmy że dane nie są interleaved czyli mamy
// (channel0: 0, 1, 2, 3 ... channel1: 0, 1, 2, 3 ... )

class TransformEasy {
	double* in;
	fftw_complex* out;
	fftw_plan plan;

public:
	const int C, N, Nout;

	TransformEasy(int C, int N, double*, fftw_complex*)
	: C(C), N(N), Nout(N/2+1) {
		in = (double*) fftw_malloc(N * sizeof(double));
		out = (fftw_complex*) fftw_malloc(Nout * sizeof(fftw_complex));
		plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_FLAGS);
	}

	~TransformEasy(void) {
		fftw_destroy_plan(plan);
		fftw_free(in);
		fftw_free(out);
		fftw_cleanup();
	}

	void compute(double* input, fftw_complex* output) {
		for (int c=0; c<C; ++c) {
			memcpy(in, input+c*N, N*sizeof(double));
			fftw_execute(plan);
			memcpy(output+c*Nout, out, Nout*sizeof(fftw_complex));
		}
	}
};

class TransformNoCopy {
	fftw_plan plan;

public:
	const int C, N, Nout;

	TransformNoCopy(int C, int N, double* input, fftw_complex* output)
	: C(C), N(N), Nout(N/2+1) {
		plan = fftw_plan_dft_r2c_1d(N, input, output, FFTW_FLAGS);
	}

	~TransformNoCopy(void) {
		fftw_destroy_plan(plan);
		fftw_cleanup();
	}

	void compute(double* input, fftw_complex* output) {
		for (int c=0; c<C; ++c) {
			fftw_execute_dft_r2c(plan, input+c*N, output+c*Nout);
		}
	}
};

class TransformMany {
	fftw_plan plan;

public:
	const int C, N, Nout;

	TransformMany(int C, int N, double* input, fftw_complex* output)
	: C(C), N(N), Nout(N/2+1) {
		plan = fftw_plan_many_dft_r2c(1, &N, C,
			input, &N,
			1, N,
			output, &Nout,
			1, Nout,
			FFTW_FLAGS
		);
	}

	~TransformMany(void) {
		fftw_destroy_plan(plan);
		fftw_cleanup();
	}

	void compute(double*, fftw_complex*) {
		fftw_execute(plan);
	}
};

class TransformKFR {
public:
	const int C, N, Nout;

private:
	kfr::dft_plan_real<double> plan;
	double* in;
	kfr::complex<double>* out;
	kfr::u8* temp;

public:
	TransformKFR(int C, int N, double*, fftw_complex*)
	: C(C), N(N), Nout(N/2+1), plan(N)
	{
		in = (double*) fftw_malloc(N * sizeof(double));
		out = (kfr::complex<double>*) fftw_malloc(Nout * sizeof(fftw_complex));
		temp = new kfr::u8[plan.temp_size];
	}

	~TransformKFR(void)
	{
		fftw_free(in);
		fftw_free(out);
		delete [] temp;
	}

	void compute(double* input, fftw_complex* output) {
		for (int c=0; c<C; ++c) {
			memcpy(in, input+c*N, N*sizeof(double));
			plan.execute(out, in, temp);
			memcpy(output+c*Nout, out, Nout*sizeof(fftw_complex));
		}
	}
};


class TransformOpenMP {
	fftw_plan plan;

public:
	const int C, N, Nout;

	TransformOpenMP(int C, int N, double* input, fftw_complex* output)
	: C(C), N(N), Nout(N/2+1) {
		plan = fftw_plan_dft_r2c_1d(N, input, output, FFTW_FLAGS);
	}

	~TransformOpenMP(void) {
		fftw_destroy_plan(plan);
		fftw_cleanup();
	}

	void compute(double* input, fftw_complex* output) const {
		#pragma omp parallel for schedule(static,1)
		for (int c=0; c<C; ++c) {
			fftw_execute_dft_r2c(plan, input+c*N, output+c*Nout);
		}
	}
};

class TransformSHM {
	static const int PROCESSES = 8;

	double *in, *all_in;
	fftw_complex *out, *all_out;
	fftw_plan plan;

	int semaphores;
	int input_id;
	int output_id;
	pid_t children[PROCESSES];

	static void forked(int N, int C, int process, int semaphores, double* all_in, fftw_complex* all_out) {
		sembuf op;
		const int Nout = N/2+1;

		// child process
		double* in = (double*) fftw_malloc(N*sizeof(double));
		fftw_complex* out = (fftw_complex*) fftw_malloc(Nout*sizeof(fftw_complex));
		fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_FLAGS);

		for (int c=process; c<C; c+=PROCESSES) {
			// decrease input semaphore by 1
			op.sem_num = 0;
			op.sem_op = -1;
			op.sem_flg = 0;
			semop(semaphores, &op, 1);
			
			// compute
			memcpy(in, all_in+c*N, N*sizeof(double));
			fftw_execute(plan);
			memcpy(all_out+c*Nout, out, Nout*sizeof(fftw_complex));

			// increase output semaphore by 1
			op.sem_num = 1;
			op.sem_op = +1;
			op.sem_flg = 0;
			semop(semaphores, &op, 1);
		}
	}

public:
	const int C, N, Nout;

	TransformSHM(int C, int N, double*, fftw_complex*)
	: C(C), N(N), Nout(N/2+1) {
		semaphores = semget(IPC_PRIVATE, 2, IPC_CREAT);
		input_id = shmget(IPC_PRIVATE, N*C*sizeof(double), IPC_CREAT);
		output_id = shmget(IPC_PRIVATE, Nout*C*sizeof(fftw_complex), IPC_CREAT);
		all_in = (double*) fftw_malloc(N*C*sizeof(double));
		shmat(input_id, all_in, 0);
		all_out = (fftw_complex*) fftw_malloc(Nout*C*sizeof(fftw_complex));
		shmat(output_id, all_out, 0);

		for (int process=1; process<PROCESSES; ++process) {
			pid_t child = fork();
			if (child) {
				children[process] = child;
			} else {
				forked(N, C, process, semaphores, all_in, all_out);
				exit(0);
			}
		}

		// master process
		in = (double*) fftw_malloc(N*sizeof(double));
		out = (fftw_complex*) fftw_malloc(Nout*sizeof(fftw_complex));
		plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_FLAGS);
	}

	~TransformSHM(void) {
		fftw_destroy_plan(plan);
		fftw_free(in);
		fftw_free(out);
		for (int process=1; process<PROCESSES; ++process) {
			kill(children[process], SIGTERM);
		}
		fftw_cleanup();
	}

	void compute(double* input, fftw_complex* output) {
		sembuf op;
		
		memcpy(all_in, input, N*C*sizeof(double));

		// increase input semaphore
		op.sem_num = 0;
		op.sem_op = +PROCESSES;
		op.sem_flg = 0;
		semop(semaphores, &op, 1);

		// compute my own part
		for (int c=0; c<C; c+=PROCESSES) {
			memcpy(in, all_in+c*N, N*sizeof(double));
			fftw_execute(plan);
			memcpy(all_out+c*Nout, out, Nout*sizeof(fftw_complex));

			// increase output semaphore by 1
			op.sem_num = 1;
			op.sem_op = +1;
			op.sem_flg = 0;
			semop(semaphores, &op, 1);
		}

		// wait for all results
		op.sem_num = 1;
		op.sem_op = -PROCESSES;
		op.sem_flg = 0;
		semop(semaphores, &op, 1);

		memcpy(output, all_out, Nout*C*sizeof(fftw_complex));
	}
};

template<class Transform>
long long benchmark(int repeats, int C, int N) {
	const int Nout = N/2+1;
	const int Ntotal = N * C;
	double* data = (double*) fftw_malloc(N * C * sizeof(double));
	fftw_complex* output = (fftw_complex*) fftw_malloc(Nout * C * sizeof(fftw_complex));
	Transform transform(C, N, data, output);

	long long time = 0;
	struct timespec t0, t1;
	for (int r=0; r<repeats; ++r) {
		for (int i=0; i<Ntotal; ++i) {
			data[i] = rand() / (double) RAND_MAX - 0.5;
		}
		clock_gettime(CLOCK_MONOTONIC, &t0);
		transform.compute(data, output);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		time += (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
	}

	free(output);
	free(data);
	return time;
}

/*
struct message_input {
	long mtype = 2;
	union {
		char text[1];
		struct {
			long channel;
			double data[0];
		};
	};
};

struct message_output {
	long mtype = 3;
	union {
		char text[0];
		struct {
			long channel;
			fftw_complex data[0];
		};
	};
};

class TransformMSQ {
	static const int PROCESSES = 8;

	double *in;
	fftw_complex *out;
	fftw_plan plan;

	int queue;
	pid_t children[PROCESSES];

	static void forked(int N, int queue) {
		const int Nout = N/2+1;

		// child process
		message_input* m_input = (message_input*) fftw_malloc(offsetof(message_input, data) + N*sizeof(double));
		message_output* m_output = (message_output*) fftw_malloc(offsetof(message_output, data) + Nout*sizeof(fftw_complex));
		fftw_plan plan = fftw_plan_dft_r2c_1d(N, m_input->data, m_output->data, FFTW_FLAGS);
		printf("receive?\n");
		while (msgrcv(queue, &m_input, 7*sizeof(long)+N*sizeof(double), 2, 0) > 0) {
			printf("before\n");
			fftw_execute(plan);
			printf("after\n");
			m_output->channel = m_input->channel;
			msgsnd(queue, &m_output, 7*sizeof(long)+Nout*sizeof(fftw_complex), 0);
		}
		printf("after loop\n");
	}

public:
	const int C, N, Nout;

	TransformMSQ(int C, int N, double*, fftw_complex*)
	: C(C), N(N), Nout(N/2+1) {
		queue = msgget(IPC_PRIVATE, IPC_CREAT);
		for (int process=1; process<PROCESSES; ++process) {
			pid_t child = fork();
			if (child) {
				children[process] = child;
			} else {
				forked(N, queue);
				exit(0);
			}
		}

		// master process
		in = (double*) fftw_malloc(N*sizeof(double));
		out = (fftw_complex*) fftw_malloc(Nout*sizeof(fftw_complex));
		plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_FLAGS);
	}

	~TransformMSQ(void) {
		fftw_destroy_plan(plan);
		fftw_free(in);
		fftw_free(out);
		for (int process=1; process<PROCESSES; ++process) {
			kill(children[process], SIGTERM);
		}
		fftw_cleanup();
	}

	void compute(double* input, fftw_complex* output) {
		message_input* m_input = (message_input*) fftw_malloc(sizeof(long) + N*sizeof(double));
		message_output* m_output = (message_output*) fftw_malloc(sizeof(long) + Nout*sizeof(fftw_complex));

		for (int c=0; c<C; ++c) {
			m_input->mtype = 2;
			m_input->channel = c;
			memcpy(&m_input->data, input+c*N, N*sizeof(double));
			msgsnd(queue, m_input, 7*sizeof(long)+N*sizeof(double), 0);
		}
		for (int c=0; c<C; ++c) {
			m_output->mtype = 3;
			msgrcv(queue, m_output, 7*sizeof(long)+Nout*sizeof(fftw_complex), 3, 0);
			memcpy(output+m_output->channel*Nout, &m_output->data, Nout*sizeof(fftw_complex));
		}
	}
};
*/

/*
long long transform_single(int repeats, int C, int N) {
	int Nout = N/2+1;
	double* data = (double*) fftw_malloc(N * C * sizeof(double));
	fftw_complex* output = (fftw_complex*) fftw_malloc(Nout * C * sizeof(fftw_complex));

	fftw_plan plan[C];
	for (int c=0; c<C; ++c) {
		plan[c] = fftw_plan_dft_r2c_1d(N, data+c*N, output+c*Nout, FFTW_FLAGS);
	}

	long long time = 0;
	struct timespec t0, t1;
	for (int r=0; r<repeats; ++r) {
		for (int c=0; c<C; ++c) {
			for (int i=0; i<N; ++i) {
				data[i*C+c] = drand48() - 0.5;
			}
		}
		clock_gettime(CLOCK_MONOTONIC, &t0);
		for (int c=0; c<C; ++c) {
			fftw_execute(plan[c]);
		}
		clock_gettime(CLOCK_MONOTONIC, &t1);
		time += (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
	}

	for (int c=0; c<C; ++c) {
		fftw_destroy_plan(plan[c]);
	}

	free(output);
	free(data);
	return time;
}


long long transform_multi(int repeats, int C, int N) {
	int Nout = N/2+1;
	double* data = (double*) fftw_malloc(N * C * sizeof(double));
	fftw_complex* output = (fftw_complex*) fftw_malloc(Nout * C * sizeof(fftw_complex));

	fftw_plan plan[C];
	for (int c=0; c<C; ++c) {
		plan[c] = fftw_plan_dft_r2c_1d(N, data+c*N, output+c*Nout, FFTW_FLAGS);
	}

	long long time = 0;
	struct timespec t0, t1;
	for (int r=0; r<repeats; ++r) {
		for (int c=0; c<C; ++c) {
			for (int i=0; i<N; ++i) {
				data[c*N+i] = drand48() - 0.5;
			}
		}
		clock_gettime(CLOCK_MONOTONIC, &t0);
		#pragma omp parallel for
		for (int c=0; c<C; ++c) {
			fftw_execute(plan[c]);
		}
		clock_gettime(CLOCK_MONOTONIC, &t1);
		time += (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
	}

	for (int c=0; c<C; ++c) {
		fftw_destroy_plan(plan[c]);
	}

	free(output);
	free(data);
	return time;
}

long long transform_advanced(int repeats, int C, int N) {
	int Nout = N/2+1;
	int Ntotal = N * C;
	fftw_complex* data = (fftw_complex*) fftw_malloc(C * N * sizeof(fftw_complex));
	fftw_complex* output = (fftw_complex*) fftw_malloc(C * N * sizeof(fftw_complex));

	long long time = 0;
	struct timespec t0, t1;
	
	fftw_plan plan = fftw_plan_many_dft(1, &N, C,
		data, &N,
		C, 1,
		output, &Nout,
		C, 1,
		FFTW_FORWARD, FFTW_FLAGS
	);
	for (int r=0; r<repeats; ++r) {
		for (int i=0; i<Ntotal; ++i) {
			data[i][0] = drand48() - 0.5;
			data[i][1] = drand48() - 0.5;
		}
		clock_gettime(CLOCK_MONOTONIC, &t0);
		fftw_execute(plan);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		time += (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
	}
	fftw_destroy_plan(plan);

	free(output);
	free(data);
	fftw_cleanup();
	return time;
}

double count_easy(int N) {
	int Nout = N/2+1;
	double* data = (double*) fftw_malloc(N * sizeof(double));
	fftw_complex* output = (fftw_complex*) fftw_malloc(Nout * sizeof(fftw_complex));

	long long plan_time = 0;
	struct timespec t0p, t1p;
	clock_gettime(CLOCK_MONOTONIC, &t0p);
	fftw_plan plan = fftw_plan_dft_r2c_1d(N, data, output, FFTW_FLAGS);
	clock_gettime(CLOCK_MONOTONIC, &t1p);
	plan_time += (t1p.tv_sec - t0p.tv_sec) * 1000000000LL + (t1p.tv_nsec - t0p.tv_nsec);

	long long time = 0;
	struct timespec t0, t1;
	for (int i=0; i<N; ++i) {
		data[i] = rand() / (double) RAND_MAX - 0.5;
	}
	clock_gettime(CLOCK_MONOTONIC, &t0);
	fftw_execute(plan);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	time += (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
	fftw_destroy_plan(plan);
	
	free(output);
	free(data);
	fftw_cleanup();
	return plan_time / (double) time;
}
*/

int main(void) {
	for (int n2=16; n2<=MAX; n2*=2)
//	for (int n3=n2; n3<=MAX; n3*=3)
//	for (int n5=n3; n5<=MAX; n5*=5)
	{
		printf("%6d %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf %10.6lf\n", n2,
			1.0e-9 * benchmark<TransformEasy>(REPEATS, CHANNELS, n2),
			1.0e-9 * benchmark<TransformNoCopy>(REPEATS, CHANNELS, n2),
			1.0e-9 * benchmark<TransformMany>(REPEATS, CHANNELS, n2),
			1.0e-9 * benchmark<TransformOpenMP>(REPEATS, CHANNELS, n2),
			1.0e-9 * benchmark<TransformSHM>(REPEATS, CHANNELS, n2),
			1.0e-9 * benchmark<TransformKFR>(REPEATS, CHANNELS, n2)
		);
		fflush(stdout);
	}
}
