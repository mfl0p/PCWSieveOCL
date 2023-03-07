/*

	check kernel

	Bryan Little 2/12/2023

	validates proper operation of the sieve kernel and notifies CPU if there was an error
	also computes the checksum using 64 bit integer local and global atomics

*/




#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable


__kernel void check(__global ulong * g_K, __global ulong * g_lK, __global uint * g_flag, __global uint * primecount, __global ulong * g_P, __global ulong * g_checksum) {

	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	__local ulong checksum;
	uint pcnt = primecount[0];

	if(lid == 0){
		checksum = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(gid < pcnt){

		ulong my_K = g_K[gid];
		ulong last_K = g_lK[gid];
		ulong my_P = g_P[gid];

		// add my_P and my_K to local checksum
		atom_add( &checksum, my_P+my_K );

		// should match if sieve kernel calculated from nmin to nmax correctly.
		if(my_K != last_K){
			// printf("n %u bbits1 %d r1 %llu checksum mismatch %llu vs %llu\n",my_lastN,bbits1,r1,my_K,kpos);
			// checksum mismatch, set flag
			atomic_or(&g_flag[0], 1);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid == 0){	
		// add local checksum to global
		atom_add( &g_checksum[0], checksum );
	}

	if(gid == 0){
		// add primecount to total primecount
		atom_add( &g_checksum[1], pcnt );

		// store largest kernel prime count
		if( pcnt > primecount[1] ){
			primecount[1] = pcnt;
		}
	}

}





