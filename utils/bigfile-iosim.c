#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <alloca.h>
#include <string.h>

#include <mpi.h>
#include <unistd.h>

#include <bigfile.h>
#include <bigfile-mpi.h>

int ThisTask;
int NTask;

static void 
usage() 
{
    if(ThisTask != 0) return;
    printf("usage: iosim [-A (to use aggregated IO)] [-N nfiles] [-n numwriters] [-s items] [-w nmemb] [-p path] [-m (create|update|read)] [-d (to delete fakedata afterwards)] filename\n");
    printf("Defaults: -N 1 -n NTask -s 1 -w 1 -p <dir> -m create \n");
}

static void 
info(char * fmt, ...) {

    static double t0 = -1.0;

    MPI_Barrier(MPI_COMM_WORLD);

    if(t0 < 0) t0 = MPI_Wtime();

    char * buf = alloca(strlen(fmt) + 100);
    sprintf(buf, "[%010.3f] %s", MPI_Wtime() - t0, fmt );

    if(ThisTask == 0) {
        va_list va;
        va_start(va, fmt);
        vprintf(buf, va);
        va_end(va);
    }
}

#define MODE_CREATE 0
#define MODE_READ   1
#define MODE_UPDATE 2

typedef struct log{
    double create;
    double open;
    double write;
    double read;
    double close;
} log;

static void
sim(int Nfile, int aggregated, int Nwriter, size_t size, int nmemb, char * filename, log * times, int mode)
{
    size_t localoffset = size * ThisTask / NTask;
    size_t localsize = size * (ThisTask + 1) / NTask - localoffset;

    info("Writing to `%s`\n", filename);
    info("Physical Files %d\n", Nfile);
    info("Ranks %d\n", NTask);
    info("Writers %d\n", Nwriter);
    info("Aggregated %d\n", aggregated);
    info("LocalBytes %td\n", localsize * 8);
    info("LocalSize %td\n", localsize);
    info("Nmemb %d\n", nmemb);
    info("Size %td\n", size);

    
    if(aggregated) {
        /* Use a large enough number to disable aggregated IO */
        big_file_mpi_set_aggregated_threshold(size * nmemb * 8);
    } else {
        big_file_mpi_set_aggregated_threshold(0);
    }
    BigFile bf = {0};
    BigBlock bb = {0};
    BigArray array = {0};
    BigBlockPtr ptr = {0};

    uint64_t * fakedata;
    ptrdiff_t i;
    //
    //+++++++++++++++++ Timelog variables +++++++++++++++++
    double t0, t1;
    log trank;
    trank.create = trank.open = trank.write = trank.read = trank.close = 0;
    int nel_trank = sizeof(trank) / sizeof(trank.create);
    //
    //+++++++++++++++++ END +++++++++++++++++


    if(mode == MODE_CREATE) {
        info("Creating BigFile\n");
        t0 = MPI_Wtime();
        big_file_mpi_create(&bf, filename, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        trank.create += t1 - t0;
        info("Created BigFile\n");

        info("Creating BigBlock\n");
        t0 = MPI_Wtime();
        big_file_mpi_create_block(&bf, &bb, "TestBlock", "i8", nmemb, Nfile, size, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        trank.create += t1 - t0;
        info("Created BigBlock\n");
    }  else {
        info("Opening BigFile\n");
        t0 = MPI_Wtime();
        big_file_mpi_open(&bf, filename, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        trank.open += t1 - t0;
        info("Opened BigFile\n");

        info("Opening BigBlock\n");
        t0 = MPI_Wtime();
        big_file_mpi_open_block(&bf, &bb, "TestBlock", MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        trank.open += t1 - t0;
        info("Opened BigBlock\n");
        if(bb.size != size) {
            info("Size mismatched, overriding size = %td\n", bb.size);
            size = bb.size;
            localoffset = size * ThisTask / NTask;
            localsize = size * (ThisTask + 1) / NTask - localoffset;
        }
        if(bb.nmemb != nmemb) {
            info("Size mismatched, overriding nmemb = %d\n", bb.nmemb);
            nmemb = bb.nmemb;
        }
        if(bb.Nfile != Nfile) {
            info("Nfile mismatched, overriding Nfile = %d\n", bb.Nfile );
            Nfile = bb.Nfile;
        }
    }

    fakedata = malloc(8 * localsize * nmemb);
    big_array_init(&array, fakedata, "i8", 2, (size_t[]){localsize, nmemb}, NULL);

    if(mode == MODE_CREATE || mode == MODE_UPDATE) {
        info("Initializing FakeData\n");
        for(i = 0; i < localsize; i ++) {
            int j;
            for(j = 0; j < nmemb; j ++) {
                fakedata[i * nmemb + j] = localoffset + i;
            }
        }
        info("Initialized FakeData\n");
        info("Writing BigBlock\n");
        t0 = MPI_Wtime();
        big_block_mpi_write(&bb, &ptr, &array, Nwriter, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        trank.write += t1 - t0;
        info("Written BigBlock\n");
        info("Writing took %f seconds\n", trank.write);

    }
    else {
        info("Reading BigBlock\n");

        big_block_seek(&bb, &ptr, 0);
        t0 = MPI_Wtime();
        big_block_mpi_read(&bb, &ptr, &array, Nwriter, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        trank.read += t1 - t0;
        info("Reading took %f seconds\n", trank.read);
        info("Initializing FakeData\n");
        for(i = 0; i < localsize; i ++) {
            int j;
            for(j = 0; j < nmemb; j ++) {
                //printf("%lX ", fakedata[i * nmemb + j]);
                if (fakedata[i * nmemb + j] != localoffset + i) {
                    info("data is corrupted either due to reading or writing\n");
                    abort();

                }
            }
        }
        //printf("\n");
        info("Initialized FakeData\n");
    }

    info("Closing BigBlock\n");
    t0 = MPI_Wtime();
    big_block_mpi_close(&bb, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    trank.close += t1 - t0;
    info("Closed BigBlock\n");

    info("Closing BigFile\n");
    t0 = MPI_Wtime();
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    trank.close += t1 - t0;
    info("Closed BigFile\n");

    MPI_Gather(&trank, nel_trank, MPI_DOUBLE, times, nel_trank, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(fakedata);
}

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    
    int i;
    int ch;
    int nmemb = 1;
    int Nfile = 1;
    int Nwriter = NTask;
    int aggregated = 0;
    size_t size = 1024;
    char * filename = alloca(1500);
    char * path = "";
    char * postfix = alloca(1500);
    int mode = MODE_CREATE;
    int delfiles = 0;
    char * buffer = alloca(1500);
    //+++++++++++++++++ Timelog +++++++++++++++++
    char * timelog = alloca(1000);
    FILE *F;

    log * times = malloc(sizeof(log) * NTask);

    while(-1 != (ch = getopt(argc, argv, "hN:n:s:w:p:m:dA"))) {
        switch(ch) {
            case 'A':
                aggregated = 1;
                break;
            case 'd':
                delfiles = 1;
                break;
            case 'm':
                if(0 == strcmp(optarg, "read")) {
                    mode = MODE_READ;
                } else
                if(0 == strcmp(optarg, "create")) {
                    mode = MODE_CREATE;
                } else
                if(0 == strcmp(optarg, "update")) {
                    mode = MODE_UPDATE;
                } else {
                    usage();
                    goto byebye;
                }
                break;
            case 'p':
                path = optarg;
                if( path[0] == '-') {
                    usage();
                    goto byebye;
                }
                break;
            case 'w':
                if(1 != sscanf(optarg, "%d", &nmemb)) {
                    usage();
                    goto byebye;
                }
                break;
            case 'N':
                if(1 != sscanf(optarg, "%d", &Nfile)) {
                    usage();
                    goto byebye;
                }
                break;
            case 'n':
                if(1 != sscanf(optarg, "%d", &Nwriter)) {
                    usage();
                    goto byebye;
                }
                break;
            case 's':
                if(1 != sscanf(optarg, "%td", &size)) {
                    usage();
                    goto byebye;
                }
                break;
            case 'h':
            default:
                usage();
                goto byebye;
        }    
    }
    if(optind == argc) {
        usage();
        goto byebye;
    }

//+++++++++++++++++ Filename and checks of input parameters +++++++++++++++++
    sprintf(filename, "%s%s", path, argv[optind]);
    if (Nwriter > NTask) {
        info("\n\n ############## CAUTION: you chose %d ranks and %d writers! ##############\n"
             " #  If you want %d writers, allocate at least %d ranks with <mpirun -n %d> #\n"
             " ################### Can only use %d writers instead! ###################\n\n",
             NTask, Nwriter, Nwriter, Nwriter, Nwriter, NTask);
        Nwriter = NTask;
    }

//+++++++++++++++++ Starting Simulation +++++++++++++++++
    sim(Nfile, aggregated, Nwriter, size, nmemb, filename, times, mode);
//+++++++++++++++++ Deleting files if flag -d set +++++++++++++++++
    if (delfiles) {
        if(ThisTask == 0) {
            sprintf(buffer, "rm -rf %s/TestBlock", filename);
            system(buffer);
        }
    }
//+++++++++++++++++ Writing Time Log +++++++++++++++++
    sprintf(timelog, "%s/Timelog", filename);
    if (ThisTask == 0){
        F = fopen(timelog, "a+");
        if (!F){
            info("iosim.c: Couldn't open file %s for writting!\n", timelog);
        }
        else{
            fprintf(F, "# files %d ranks %d writers %d items %td nmemb %d\n",
                         Nfile, NTask, Nwriter, size, nmemb);

            fprintf(F, "# Task\tTcreate\t\tTopen\t\tTwrite\t\tTread\t\tTclose\n");
            for (i=0; i<NTask; i++) {
                fprintf(F, "%d\t%f\t%f\t%f\t%f\t%f\n",
                    i, times[i].create, times[i].open, times[i].write, times[i].read, times[i].close);
            }
        }
        fclose(F);
    }

byebye:
    free(times);
    MPI_Finalize();
    return 0;
}
