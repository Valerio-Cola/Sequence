/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * MPI version
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
#include<mpi.h>


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
 *
 */

/*
 * Function: Increment the number of pattern matches on the sequence positions
 * 	This function can be changed and/or optimized by the students
 */
void increment_matches( int pat, unsigned long *pat_found, unsigned long *pat_length, int *seq_matches ) {
	unsigned long ind;	
	for( ind=0; ind<pat_length[pat]; ind++) {
		if ( seq_matches[ pat_found[pat] + ind ] == NOT_FOUND )
			seq_matches[ pat_found[pat] + ind ] = 0;
		else
			seq_matches[ pat_found[pat] + ind ]++;
	}
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
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* Return results */
	*new_length = length;
	return pattern;
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
	/* 1.0. Init MPI before processing arguments */
	MPI_Init( &argc, &argv );
	int rank;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	/* 1.1. Check minimum number of arguments */
	if (argc < 15) {
		fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	unsigned long seq_length = atol( argv[1] );
	float prob_G = atof( argv[2] );
	float prob_C = atof( argv[3] );
	float prob_A = atof( argv[4] );
	if ( prob_G + prob_C + prob_A > 1 ) {
		fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	unsigned long seed = atol( argv[14] );

#ifdef DEBUG
	/* DEBUG: Print arguments */
	if ( rank == 0 ) {
		printf("\nArguments: seq_length=%lu\n", seq_length );
		printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A );
		printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev );
		printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev );
		printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed );
		printf("\n");
	}
#endif // DEBUG

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
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		}
	}
	free( pat_type );

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
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	
	/* 3. Start global timer */
	MPI_Barrier( MPI_COMM_WORLD );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT 
 *
 */
	int size;
	MPI_Comm_size( MPI_COMM_WORLD, &size );
	
	// Solo il rank 0 genera la sequenza e la invia a tutti gli altri processi

	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	if(rank == 0){
		random = rng_new( seed );
		generate_rng_sequence( &random, prob_G, prob_C, prob_A, sequence, seq_length);
	}

	MPI_Bcast(sequence, seq_length, MPI_CHAR, 0, MPI_COMM_WORLD);

	MPI_Barrier( MPI_COMM_WORLD );

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

	// Ognuno genera il proprio seq_matches 
	int *seq_matches;
	seq_matches = (int *)malloc( sizeof(int) * seq_length );
	if ( seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	MPI_Barrier( MPI_COMM_WORLD );

	// pat_number e seq_matches inizializzati a -1
	for( ind=0; ind<pat_number; ind++) {
		pat_found[ind] = (unsigned long)NOT_FOUND;
	}
	for( lind=0; lind<seq_length; lind++) {
		seq_matches[lind] = NOT_FOUND;
	}
	MPI_Barrier( MPI_COMM_WORLD );


	// Ognuno si prende tot pattern da cercare
	int my_patterns = pat_number/size;
	int resto = pat_number % size;

	// Si tiene traccia del primo e dell'ultimo pattern
	int my_first_pattern = rank * my_patterns;
	
	if(rank == size - 1){
		my_patterns += resto;
	}

	int my_last_pattern = my_first_pattern + my_patterns -1;

	MPI_Barrier( MPI_COMM_WORLD );

	// Ogni processo cerca i pattern assegnati
	unsigned long start;
	int pat;
	for( pat = my_first_pattern; pat <= my_last_pattern; pat++ ) {

		// Cerca il pattern pattern in tutta la sequenza
		for( start=0; start <= seq_length - pat_length[pat]; start++) {
			for( lind=0; lind<pat_length[pat]; lind++) {
				if ( sequence[start + lind] != pattern[pat][lind] ) break;
			}
			if ( lind == pat_length[pat] ) {
				pat_matches++;
				pat_found[pat] = start;				
				break;
			}
		}

		if ( pat_found[pat] != (unsigned long)NOT_FOUND ) {
			increment_matches( pat, pat_found, pat_length, seq_matches );
		}
	}

	// Tutti i rank inviano i propri seq_matches al rank 0
	if(rank > 0){
		MPI_Send(seq_matches, seq_length, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	
	//MPI_Barrier( MPI_COMM_WORLD );

	// Il rank 0 si occuperà di sommare i seq_matches di tutti i rank
	// Sicuramente si può ottimizzare questa parte

	// Risolto:
	// Nota che seq_matches parte da tutti -1 e viene incrementato man mano che viene trovato un pattern
	// il problema sta nel caso in cui un pattern si ripete tra più rank e quindi viene incrementato più volte da -1 a 0 e si fotte il checksum dopo
	if (rank == 0) {
		int *local_seq_matches;
		local_seq_matches = (int *)malloc( sizeof(int) * seq_length );
		
		for (int i = 1; i < size; i++) {
			MPI_Recv(local_seq_matches, seq_length, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			for (unsigned long j = 0; j < seq_length; j++) {
				if (seq_matches[j] >= 0 && local_seq_matches[j] >= 0) {
					seq_matches[j] = seq_matches[j] + local_seq_matches[j] + 1;
				} else {
					if(seq_matches[j] == NOT_FOUND){
						seq_matches[j] = local_seq_matches[j];
					}
					
				}
			}
		}
		
		free(local_seq_matches);
	}
	MPI_Barrier( MPI_COMM_WORLD );
	
	// Ognuno invia i pattern trovati al rank 0
	//MPI_Gather(pat_found + my_first_pattern, my_patterns, MPI_UNSIGNED_LONG, pat_found, my_patterns, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	
	int send_counts[size]; // {133, 133, 134}; // Numero di elementi da inviare per ciascun rank 
	send_counts[rank] = my_patterns;

	int displs[size]; //= {0, 133, 266};
	displs[rank] = my_first_pattern;
	
	MPI_Allgather(&my_patterns, 1, MPI_INT, send_counts, 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(&my_first_pattern, 1, MPI_INT, displs, 1, MPI_INT, MPI_COMM_WORLD);

	MPI_Barrier( MPI_COMM_WORLD );

	MPI_Gatherv(pat_found + my_first_pattern, my_patterns, MPI_UNSIGNED_LONG, pat_found, send_counts, displs, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	
	// Ognuno invia al rank 0 il numero di pattern trovati sommandoli con la reduce
	int total_pat_matches = 0;
	MPI_Reduce(&pat_matches, &total_pat_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Barrier( MPI_COMM_WORLD );

	// Il rank 0 assegna il valore di total_pat_matches a pat_matches
	if (rank == 0) {
		pat_matches = total_pat_matches;
	}

	MPI_Barrier( MPI_COMM_WORLD );

	// Calcolo checksum, sequenziale e non devono dare in output gli stessi checksum
	// Poiche vengono stampati solo dal rank 0 piu giù lo facciamo fare solo a lui
	unsigned long checksum_matches = 0;
	unsigned long checksum_found = 0;
	if(rank == 0){
		for( ind=0; ind < pat_number; ind++) {
			if ( pat_found[ind] != (unsigned long)NOT_FOUND )
				checksum_found = ( checksum_found + pat_found[ind] ) % CHECKSUM_MAX;
		}
		for( lind=0; lind < seq_length; lind++) {
			if ( seq_matches[lind] != NOT_FOUND )
				checksum_matches = ( checksum_matches + seq_matches[lind] ) % CHECKSUM_MAX;
		}
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

	/* 8. Stop global time */
	MPI_Barrier( MPI_COMM_WORLD );
	ttotal = cp_Wtime() - ttotal;

	/* 9. Output for leaderboard */
	if ( rank == 0 ) {
		printf("\n");
		/* 9.1. Total computation time */
		printf("Time: %lf\n", ttotal );

		/* 9.2. Results: Statistics */
		printf("Result: %d, %lu, %lu\n\n", 
				pat_matches,
				checksum_found,
				checksum_matches );
	}
				
	/* 10. Free resources */	
	int i;
	for( i=0; i<pat_number; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );

	/* 11. End */
	MPI_Finalize();
	return 0;
}
