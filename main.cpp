/*
	PCWSieve
	Bryan Little, Feb 21 2023
	
	Search algorithm by
	Geoffrey Reynolds, 2009
	Ken Brazier, 2009
	https://github.com/Ken-g6/PSieve-CUDA/tree/redcl
	https://github.com/Ken-g6/PSieve-CUDA/tree/cw

	With contributions by
	Yves Gallot

	Required minimum OpenCL version is 2.0
	CL_TARGET_OPENCL_VERSION 200 in simpleCL.h

*/

#include <unistd.h>
#include <getopt.h>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"
#include "primesieve.h"
#include "putil.h"
#include "cl_sieve.h"

using namespace std; 


void help()
{
	printf("Program usage\n");
	printf("-p #\n");
	printf("-P #			Sieve primes -p <= p <= -P < 2^62\n");
	printf("-k #\n");
	printf("-K #			Sieve for primes k*2^n+/-1 with -k <= k <= -K < 2^32\n");
	printf("-n #\n");
	printf("-N # 			Sieve for primes k*2^n+/-1 with -n <= n <= -N < 2^32\n");
	printf("-c 			Search for Cullen/Woodall factors\n");
	printf("-s or --test		Perform self test to verify proper operation of the program.\n");
	printf("-h			Print this help\n");
        boinc_finish(EXIT_FAILURE);
}


static const char *short_opts = "p:P:k:K:n:N:csd:h";

static int parse_option(int opt, char *arg, const char *source, searchData & sd)
{
  int status = 0;

  switch (opt)
  {
    case 'p':
      status = parse_uint64(&sd.pmin,arg,3,(UINT64_C(1)<<62)-1);
      break;

    case 'P':
      status = parse_uint64(&sd.pmax,arg,4,(UINT64_C(1)<<62)-1);
      break;

    case 'k':
      status = parse_uint(&sd.kmin,arg,1,(1U<<31)-1);
      break;

    case 'K':
      status = parse_uint(&sd.kmax,arg,1,(1U<<31)-1);
      break;
      
    case 'n':
      status = parse_uint(&sd.nmin,arg,1,(1U<<31)-1);
      break;

    case 'N':
      status = parse_uint(&sd.nmax,arg,1,(1U<<31)-1);
      break;

    case 'c':
      sd.cw = true;
      fprintf(stderr,"Searching for Cullen/Woodall factors.\n");
      printf("Searching for Cullen/Woodall factors.\n");
      break;

    case 's':
      sd.test = true;
      fprintf(stderr,"Performing self test.\n");
      printf("Performing self test.\n");
      break;

    case 'd':
      break;

    case 'h':
      help();
      break;

    case '?':
      status = -3;
      break;
  }

  return status;
}

static const struct option long_opts[] = {
  {"device",  optional_argument, 0, 'd'},		// handle --device arg, but it's not used
  {"test",  no_argument, 0, 's'},
  {0,0,0,0}
};


/* Process command-line options using getopt_long().
   Non-option arguments are treated as if they belong to option zero.
   Returns the number of options processed.
 */
static int process_args(int argc, char *argv[], searchData & sd)
{
  int count = 0, ind = -1, opt;

  while ((opt = getopt_long(argc,argv,short_opts,long_opts,&ind)) != -1)
    switch (parse_option(opt,optarg,NULL,sd))
    {
      case 0:
        ind = -1;
        count++;
        break;

      case -1:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: invalid argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: invalid argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
     	  printf("%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: invalid argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      case -2:
        /* If ind is unchanged then this is a short option, otherwise long. */
        if (ind == -1){
          printf("%s: out of range argument -%c %s\n",argv[0],opt,optarg);
          fprintf(stderr,"%s: out of range argument -%c %s\n",argv[0],opt,optarg);
	}
        else{
          printf("%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
          fprintf(stderr,"%s: out of range argument --%s %s\n",argv[0],long_opts[ind].name,optarg);
	}
        boinc_finish(EXIT_FAILURE);

      default:
        printf("unknown command line argument\n");
        boinc_finish(EXIT_FAILURE);
    }

  while (optind < argc)
    switch (parse_option(0,argv[optind],NULL,sd))
    {
      case 0:
        optind++;
        count++;
        break;

      case -1:
        fprintf(stderr,"%s: invalid non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      case -2:
        fprintf(stderr,"%s: out of range non-option argument %s\n",argv[0],argv[optind]);
        boinc_finish(EXIT_FAILURE);

      default:
        boinc_finish(EXIT_FAILURE);
    }


  return count;
}


#ifdef _WIN32
double getSysOpType()
{
    double ret = 0.0;
    NTSTATUS(WINAPI *RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        ret = (double)osInfo.dwMajorVersion;
    }
    return ret;
}
#endif


int main(int argc, char *argv[])
{ 
	sclHard hardware;
	searchData sd;

	primesieve_set_num_threads(1);

//	_putenv_s("CUDA_CACHE_DISABLE", "1");

        // Initialize BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
        options.normal_thread_priority = true;
        boinc_init_options(&options);

	fprintf(stderr, "\nPCWSieve version %s by Bryan Little\n",VERS);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	if(boinc_is_standalone()){
		printf("PCWSieve version %s by Bryan Little\n",VERS);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");

	}

        // Print out cmd line for diagnostics
        fprintf(stderr, "Command line: ");
        for (int i = 0; i < argc; i++)
        	fprintf(stderr, "%s ", argv[i]);
        fprintf(stderr, "\n");


	process_args(argc,argv,sd);


	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context ctx;
	cl_command_queue queue;
	cl_int err = 0;

	int retval = 0;
	retval = boinc_get_opencl_ids(argc, argv, 0, &device, &platform);
	if (retval) {
		if(boinc_is_standalone()){
			printf("init_data.xml not found, using device 0.\n");

			err = clGetPlatformIDs(1, &platform, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetPlatformIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetPlatformIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetDeviceIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetDeviceIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
		}
		else{
			fprintf(stderr, "Error: boinc_get_opencl_ids() failed with error %d\n", retval );
			exit(EXIT_FAILURE);
		}
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: clCreateContext() returned %d\n", err);
        	exit(EXIT_FAILURE); 
   	}

	cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };

	queue = clCreateCommandQueueWithProperties(ctx, device, qp, &err);
	if(err != CL_SUCCESS) { 
		fprintf(stderr, "Error: Creating Command Queue. (clCreateCommandQueueWithProperties) returned %d\n", err );
		exit(EXIT_FAILURE);
    	}

	hardware.platform = platform;
	hardware.device = device;
	hardware.queue = queue;
	hardware.context = ctx;

 	char device_string0[1024];
 	char device_string1[1024];
 	char device_string2[1024];
	cl_uint CUs;

	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string0), &device_string0, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string1), &device_string1, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string2), &device_string2, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CUs, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}

	fprintf(stderr, "GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_string0, device_string1, device_string2, CUs);
	if(boinc_is_standalone()){
		printf("GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_string0, device_string1, device_string2, CUs);
	}

	// check vendor and normalize compute units
	// kernel size will be determined by profiling so this doesn't have to be accurate.
	int computeunits = (int)CUs;
	char intel_s[] = "Intel";
	char arc_s[] = "Arc";
	char nvidia_s[] = "NVIDIA";	

	if(strstr((char*)device_string1, (char*)nvidia_s) != NULL){

#ifdef _WIN32
		// pascal or newer gpu on windows 10,11 allows long kernel runtimes without screen refresh issues

		float winVer = (float)getSysOpType();

		if(winVer >= 10.0f && !sd.compute){

		 	cl_uint ccmajor;
			err = clGetDeviceInfo(hardware.device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(ccmajor), &ccmajor, NULL);
			if ( err != CL_SUCCESS ) {
				printf( "Error checking device compute capability\n" );
				fprintf(stderr, "Error checking device compute capability\n");
				exit(EXIT_FAILURE);
			}

			if(ccmajor >= 6){
				sd.compute = true;
			}
		}

#else
		// linux
		// list of gpus without video output.  datacenter or mining cards.
		char dc0[] = "H100";
		char dc1[] = "A100";
		char dc2[] = "V100";
		char dc3[] = "T4";
		char dc4[] = "P106";
		char dc5[] = "P104";
		char dc6[] = "P102";
		char dc7[] = "P100";
		char dc8[] = "CMP";
		char dc9[] = "A2";
		char dc10[] = "A10";
		char dc11[] = "A16";
		char dc12[] = "A30";
		char dc13[] = "A40";

		if(	strstr((char*)device_string0, (char*)dc0) != NULL
			|| strstr((char*)device_string0, (char*)dc1) != NULL
			|| strstr((char*)device_string0, (char*)dc2) != NULL
			|| strstr((char*)device_string0, (char*)dc3) != NULL
			|| strstr((char*)device_string0, (char*)dc4) != NULL
			|| strstr((char*)device_string0, (char*)dc5) != NULL
			|| strstr((char*)device_string0, (char*)dc6) != NULL
			|| strstr((char*)device_string0, (char*)dc7) != NULL
			|| strstr((char*)device_string0, (char*)dc8) != NULL
			|| strstr((char*)device_string0, (char*)dc9) != NULL
			|| strstr((char*)device_string0, (char*)dc10) != NULL
			|| strstr((char*)device_string0, (char*)dc11) != NULL
			|| strstr((char*)device_string0, (char*)dc12) != NULL
			|| strstr((char*)device_string0, (char*)dc13) != NULL){
			sd.compute = true;
		}

#endif
	}
	// Intel integrated Graphics
	else if( strstr((char*)device_string1, (char*)intel_s) != NULL
		 && 
		 strstr((char*)device_string1, (char*)arc_s) == NULL ){

		computeunits /= 20;

                fprintf(stderr,"Detected Intel integrated graphics\n");	
	}
	// AMD, Intel Arc GPUs
        else{
		computeunits /= 2;
        }

	if(computeunits < 1){
		computeunits = 1;
	}

	sd.computeunits = computeunits;

	
	if(sd.test == true){
		run_test(hardware, sd);

	}
	else{
		cl_sieve(hardware, sd);
	}


        sclReleaseClHard(hardware);

	boinc_finish(EXIT_SUCCESS);

	return 0; 
} 

