/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.3 
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the CUDA assignment versions */
#include<cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION( call )	{ cudaError_t check = call; if ( check != cudaSuccess ) fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }
#define CUDA_CHECK_KERNEL( )	{ cudaError_t check = cudaGetLastError(); if ( check != cudaSuccess ) fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }

/* Arbitrary value to indicate that no matches are found */
#define	NOT_FOUND	-1

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX	65535


/* 
 * Utils: Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Random generator
 */
#include "rng.c"


/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
/* ADD KERNELS AND OTHER FUNCTIONS HERE */

/*
 * Function: Increment the number of pattern matches on the sequence positions
 * 	This function can be changed and/or optimized by the students
 */
__device__ void increment_matches( int pat, unsigned long *pat_found, unsigned long *pat_length, int *seq_matches ) {
	unsigned long ind;
	//__syncthreads();	
	for( ind=0; ind<pat_length[pat]; ind++) {
		
			atomicAdd(&seq_matches[ pat_found[pat] + ind ], 1);
			//seq_matches[ pat_found[pat] + ind ] ++;
	}
	//__syncthreads();
}

__global__ void sequencer(unsigned long *g_seq_length, int *g_pat_number, char *g_sequence, unsigned long *d_pat_length, char **d_pattern, int *g_seq_matches, int *g_pat_matches, unsigned long *g_pat_found) { 
    unsigned long start;
    int pat;
    unsigned long lind;
    /* Se vogliamo fare che ogni thread ha una sola sequenza da cercare
            Questo primo for è inutile
            il nostro thread int i = blockIdx.x * blockDim.x + threadIdx.x; (il prof ha fatto una cosa simile nella moltiplicazione tra vettori)
            verifichiamo che intanto sia uno di quelli che deve lavorare indipendentemente dal blocco:  i < g_pat_number
            e poi utilizziamo il suo indice come variabile pat  quindi pat = i nella dichiarazione.
            Io ho intanto pensato a questa implementazione se vuoi fare in unaltro modo non cancellare questi commenti.

            noi possiamo trovare il (o i) pattern da cercare basandoci sull'indice del thread
            se facciamo che ogni thread cerca solo un pattern dobbiamo organizzare i thread nel blocco in un certo modo
    */

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Total patterns to find: %d\n", *g_pat_number);
    if (tid < *g_pat_number) {
        //printf("Thread %d is working\n", tid);
        // esegui il lavoro solo se l'ID del thread è valido
        pat = tid;
        /* 5.1. For each posible starting position */
        for( start=0; start <= *g_seq_length - d_pat_length[pat]; start++) {
            /* 5.1.1. For each pattern element */
            for( lind=0; lind<d_pat_length[pat]; lind++) {

                /* Stop this test when different nucleotids are found */
                if ( g_sequence[start + lind] != d_pattern[pat][lind] ) break;
            }
            /* 5.1.2. Check if the loop ended with a match */
            if ( lind == d_pat_length[pat] ) {
                //printf("Pattern %d found at position %lu Tid: %d lind: %lu pat_lenght: %lu\n", pat, start, tid, lind, d_pat_length[pat]);
                // qua ho tolto il & perché era un errore di indirizzamento
                atomicAdd(g_pat_matches, 1);
                //printf("Thread %d: Total pattern matches: %d\n", tid, atomicAdd(g_pat_matches, 0));
                // qua invece ho castato le variabili in unsigned long long non so perché prima non andasse bene
                atomicExch((unsigned long long*)&g_pat_found[pat], (unsigned long long)start);
                break;
            }
        }

        /* 5.2. Pattern found */
        if ( g_pat_found[pat] != (unsigned long)NOT_FOUND ) {
            /* 4.2.1. Increment the number of pattern matches on the sequence positions */
            increment_matches( pat, g_pat_found, d_pat_length, g_seq_matches );
        }
		//__syncthreads();
	}

}

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate( rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length ) {

	/* Random length */
	unsigned long length = (unsigned long)rng_next_normal( random, (double)pat_rng_length_mean, (double)pat_rng_length_dev );
	if ( length > seq_length ) length = seq_length;
	if ( length <= 0 ) length = 1;

	/* Allocate pattern */
	char *pattern = (char *)malloc( sizeof(char) * length );
	if ( pattern == NULL ) {
		fprintf(stderr,"\n-- Error allocating a pattern of size: %lu\n", length );
		exit( EXIT_FAILURE );
	}

	/* Return results */
	*new_length = length;
	return pattern;
}

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence( rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
	unsigned long ind; 
	for( ind=0; ind<length; ind++ ) {
		double prob = rng_next( random );
		if( prob < prob_G ) seq[ind] = 'G';
		else if( prob < prob_C ) seq[ind] = 'C';
		else if( prob < prob_A ) seq[ind] = 'A';
		else seq[ind] = 'T';
	}
}

/*
 * Function: Copy a sample of the sequence
 */
void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Copy sample */
	unsigned long ind; 
	for( ind=0; ind<length; ind++ )
		pattern[ind] = sequence[ind+location];
}

/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence( rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length ) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Regenerate sample */
	rng_t local_random = random_seq;
	rng_skip( &local_random, location );
	generate_rng_sequence( &local_random, prob_G, prob_C, prob_A, pattern, length);
}


/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
	fprintf(stderr,"\n");
}



/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	/* 0. Default output and error without buffering, forces to write immediately */
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	/* 1. Read scenary arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 15) {
		fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	unsigned long seq_length = atol( argv[1] );
	float prob_G = atof( argv[2] );
	float prob_C = atof( argv[3] );
	float prob_A = atof( argv[4] );
	if ( prob_G + prob_C + prob_A > 1 ) {
		fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}
	prob_C += prob_G;
	prob_A += prob_C;

	int pat_rng_num = atoi( argv[5] );
	unsigned long pat_rng_length_mean = atol( argv[6] );
	unsigned long pat_rng_length_dev = atol( argv[7] );
	
	int pat_samp_num = atoi( argv[8] );
	unsigned long pat_samp_length_mean = atol( argv[9] );
	unsigned long pat_samp_length_dev = atol( argv[10] );
	unsigned long pat_samp_loc_mean = atol( argv[11] );
	unsigned long pat_samp_loc_dev = atol( argv[12] );

	char pat_samp_mix = argv[13][0];
	if ( pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M' ) {
		fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	unsigned long seed = atol( argv[14] );

#ifdef DEBUG
	/* DEBUG: Print arguments */
	printf("\nArguments: seq_length=%lu\n", seq_length );
	printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A );
	printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev );
	printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev );
	printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed );
	printf("\n");
#endif // DEBUG

        CUDA_CHECK_FUNCTION( cudaSetDevice(0) );

	/* 2. Initialize data structures */
	/* 2.1. Skip allocate and fill sequence */
	rng_t random = rng_new( seed );
	rng_skip( &random, seq_length );

	/* 2.2. Allocate and fill patterns */
	/* 2.2.1 Allocate main structures */
	int pat_number = pat_rng_num + pat_samp_num;
	unsigned long *pat_length = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	char **pattern = (char **)malloc( sizeof(char*) * pat_number );
	if ( pattern == NULL || pat_length == NULL ) {
		fprintf(stderr,"\n-- Error allocating the basic patterns structures for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}

	/* 2.2.2 Allocate and initialize ancillary structure for pattern types */
	int ind;
	unsigned long lind;
	#define PAT_TYPE_NONE	0
	#define PAT_TYPE_RNG	1
	#define PAT_TYPE_SAMP	2
	char *pat_type = (char *)malloc( sizeof(char) * pat_number );
	if ( pat_type == NULL ) {
		fprintf(stderr,"\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_NONE;

	/* 2.2.3 Fill up pattern types using the chosen mode */
	switch( pat_samp_mix ) {
	case 'A':
		for( ind=0; ind<pat_rng_num; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		break;
	case 'B':
		for( ind=0; ind<pat_samp_num; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		break;
	default:
		if ( pat_rng_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		}
		else if ( pat_samp_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		}
		else if ( pat_rng_num < pat_samp_num ) {
			int interval = pat_number / pat_rng_num;
			for( ind=0; ind<pat_number; ind++ ) 
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_RNG;
				else pat_type[ind] = PAT_TYPE_SAMP;
		}
		else {
			int interval = pat_number / pat_samp_num;
			for( ind=0; ind<pat_number; ind++ ) 
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_SAMP;
				else pat_type[ind] = PAT_TYPE_RNG;
		}
	}

	/* 2.2.4 Generate the patterns */
	for( ind=0; ind<pat_number; ind++ ) {
		if ( pat_type[ind] == PAT_TYPE_RNG ) {
			pattern[ind] = pattern_allocate( &random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind] );
			generate_rng_sequence( &random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind] );
		}
		else if ( pat_type[ind] == PAT_TYPE_SAMP ) {
			pattern[ind] = pattern_allocate( &random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind] );
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
			rng_t random_seq_orig = rng_new( seed );
			generate_sample_sequence( &random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#else
			copy_sample_sequence( &random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#endif
		}
		else {
			fprintf(stderr,"\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind );
			exit( EXIT_FAILURE );
		}
	}
	free( pat_type );

	/* Allocate and move the patterns to the GPU */
	unsigned long *d_pat_length;
	char **d_pattern;
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pat_length, sizeof(unsigned long) * pat_number ) );
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pattern, sizeof(char *) * pat_number ) );

	char **d_pattern_in_host = (char **)malloc( sizeof(char*) * pat_number );
	if ( d_pattern_in_host == NULL ) {
		fprintf(stderr,"\n-- Error allocating the patterns structures replicated in the host for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	unsigned long long int bytes = 0;
	for( ind=0; ind<pat_number; ind++ ) {
		printf("Allocated Size: %llu bytes\n", bytes);
		CUDA_CHECK_FUNCTION( cudaMalloc( &(d_pattern_in_host[ind]), sizeof(char *) * pat_length[ind] ) );
        	CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern_in_host[ind], pattern[ind], pat_length[ind] * sizeof(char), cudaMemcpyHostToDevice ) );
	}
	CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern, d_pattern_in_host, pat_number * sizeof(char *), cudaMemcpyHostToDevice ) );

	/* Avoid the usage of arguments to take strategic decisions
	 * In a real case the user only has the patterns and sequence data to analize
	 */
	argc = 0;
	argv = NULL;
	pat_rng_num = 0;
	pat_rng_length_mean = 0;
	pat_rng_length_dev = 0;
	pat_samp_num = 0;
	pat_samp_length_mean = 0;
	pat_samp_length_dev = 0;
	pat_samp_loc_mean = 0;
	pat_samp_loc_dev = 0;
	pat_samp_mix = '0';

	/* 2.3. Other result data and structures */
	int pat_matches = 0;

	/* 2.3.1. Other results related to patterns */
	unsigned long *pat_found;
	pat_found = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	if ( pat_found == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux pattern structure for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	
	/* 3. Start global timer */
        CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
	/* 2.1. Allocate and fill sequence */
	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
	}

	random = rng_new( seed );
	generate_rng_sequence( &random, prob_G, prob_C, prob_A, sequence, seq_length);

#ifdef DEBUG
	/* DEBUG: Print sequence and patterns */
	printf("-----------------\n");
	printf("Sequence: ");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( "%c", sequence[lind] );
	printf("\n-----------------\n");
	printf("Patterns: %d ( rng: %d, samples: %d )\n", pat_number, pat_rng_num, pat_samp_num );
	int debug_pat;
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( "Pat[%d]: ", debug_pat );
		for( lind=0; lind<pat_length[debug_pat]; lind++ ) 
			printf( "%c", pattern[debug_pat][lind] );
		printf("\n");
	}
	printf("-----------------\n\n");
#endif // DEBUG


	/* 2.3.2. Other results related to the main sequence */
	int *seq_matches;
	seq_matches = (int *)malloc( sizeof(int) * seq_length );
	if ( seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
	}

	/* 4. Initialize ancillary structures */
	for( ind=0; ind<pat_number; ind++) {
		pat_found[ind] = (unsigned long)NOT_FOUND;
	}
	for( lind=0; lind<seq_length; lind++) {
		seq_matches[lind] = NOT_FOUND;
	}

	// Variabili che non verranno modificate le sposto nella constant memory
	// NOTA: d_pattern e d_pat_lenght gia allocati in GPU
	unsigned long *g_seq_length;
	int *g_pat_number;
	char *g_sequence;

	/* 
	alla fine ho lasciato tutto in global memory per due motivi: perché la memoria globale è cached 
	e poi perché pattern e pat_length sono già in GPU allocati in global nella sezione di codice che non si può modificare,
	quindi lo stesso ragionamento si applica anche sequence
	inoltre con la clausola __constant__ vai ad usare già la shared memory (secondo le slide del prof)
	*/

	//cudaMemcpyToSymbol(g_seq_length, &seq_length, sizeof(unsigned long));
	//cudaMemcpyToSymbol(g_sequence, sequence, seq_length * sizeof(char));
	//cudaMemcpyToSymbol(g_pat_number, &pat_number, sizeof(int));


	// Necessariamente nella globale della GPU nel caso ci siano più blocchi che devono modificare
	int *g_seq_matches;
	int *g_pat_matches;
	unsigned long *g_pat_found;


	cudaMalloc(&g_seq_matches, seq_length * sizeof(int));
	cudaMalloc(&g_pat_matches, sizeof(int));
	cudaMalloc(&g_pat_found, pat_number * sizeof(unsigned long));
	cudaMalloc(&g_seq_length, sizeof(unsigned long));
	cudaMalloc(&g_pat_number, sizeof(int));
	cudaMalloc(&g_sequence, seq_length * sizeof(char));

	cudaMemcpy(g_seq_length, &seq_length, sizeof(unsigned long), cudaMemcpyHostToDevice);
	cudaMemcpy(g_pat_number, &pat_number, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sequence, sequence, seq_length * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(g_seq_matches, seq_matches, seq_length * sizeof(int), cudaMemcpyHostToDevice);
	int init_value = 0;
	cudaMemcpy(g_pat_matches, &init_value, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(g_pat_found, pat_found, pat_number * sizeof(unsigned long), cudaMemcpyHostToDevice);


	// Ponendo di avere 256 thread per blocco potremmo fare ceil(pat_number/256.0)
	// cosi da calcolare il numero di blocchi necessari per dividere le sequenze da cercare tra i thread
	// Potremmo quindi fare che ogni thread cerca una sola sequenza?

	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_length, pat_length, pat_number * sizeof(unsigned long), cudaMemcpyHostToDevice));

	/*
	Facciamo con 1024 thread per blocco, questo perché il massimo numero di thread per SM nell'architettura Turing è 1024 (max 32 warp per SM, ogni warp è da 32 threads)
	per calcolare il numero di blocchi lo facciamo con ceil(pat_number/1024.0), possiamo pure provare con altre grandezze di blocchi (max numero di blocchi per SM è 16)
	noi stiamo facendo che ogni thread ha una sequenza da vedere, ma se ci sono meno di 1024 sequenze che succede?
	Aggiunto: succede che si genera solo un blocco e lavorano solo tot thread, il resto rimane inutilizzato
	*/

	// Indicativa per i test
	sequencer<<<ceil(pat_number/1024.0), 1024>>>(g_seq_length, g_pat_number, g_sequence, d_pat_length, d_pattern, g_seq_matches, g_pat_matches, g_pat_found);

	cudaDeviceSynchronize();

	// Riporto le variabili, che il kernel ha modificato, utilizzate dal checksum nell'host
	
	cudaMemcpy(seq_matches, g_seq_matches, seq_length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&pat_matches, g_pat_matches, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pat_found, g_pat_found, pat_number * sizeof(unsigned long), cudaMemcpyDeviceToHost);


	cudaFree(g_seq_matches);
	cudaFree(g_pat_matches);
	cudaFree(g_pat_found);
	cudaFree(d_pattern);
	cudaFree(d_pat_length);

	cudaDeviceSynchronize();

	/* Debug: Print seq_matches array */
	printf("Sequence matches: ");
	for (lind = 0; lind < seq_length; lind++) {
		printf("%d ", seq_matches[lind]);
	}
	printf("\n");
	
	/* 7. Check sums */
	unsigned long checksum_matches = 0;
	unsigned long checksum_found = 0;
	for( ind=0; ind < pat_number; ind++) {
		if ( pat_found[ind] != (unsigned long)NOT_FOUND )
			checksum_found = ( checksum_found + pat_found[ind] ) % CHECKSUM_MAX;
	}
	for( lind=0; lind < seq_length; lind++) {
		if ( seq_matches[lind] != NOT_FOUND )
			checksum_matches = ( checksum_matches + seq_matches[lind] ) % CHECKSUM_MAX;
	}

#ifdef DEBUG
	/* DEBUG: Write results */
	printf("-----------------\n");
	printf("Found start:");
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( " %lu", pat_found[debug_pat] );
	}
	printf("\n");
	printf("-----------------\n");
	printf("Matches:");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( " %d", seq_matches[lind] );
	printf("\n");
	printf("-----------------\n");
#endif // DEBUG

	/* Free local resources */	

	free( sequence );
	free( seq_matches );

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 8. Stop global timer */
        CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	ttotal = cp_Wtime() - ttotal;

	/* 9. Output for leaderboard */
	printf("\n");
	/* 9.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 9.2. Results: Statistics */
	printf("Result: %d, %lu, %lu\n\n", 
			pat_matches,
			checksum_found,
			checksum_matches );
		
	/* 10. Free resources */	
	int i;
	for( i=0; i<pat_number; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );

	/* 11. End */
	return 0;
}
