/*

	clearresult kernel

	Bryan Little 2/12/2023

	Clears results

*/


__kernel void clearresult(__global uint *flag, __global uint *factorcount, __global ulong *checksum, __global uint *primecount){

	int i = get_global_id(0);

	if(i == 0){
		checksum[0] = 0;	// checksum
		checksum[1] = 0;	// total prime count between checkpoints
		factorcount[0] = 0;	// # of factors found
		flag[0] = 0;		// set to 1 if there is a gpu checksum error
		primecount[1]=0;	// keep track of largest kernel prime count
	}

}

