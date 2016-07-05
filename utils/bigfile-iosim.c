#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
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
    printf("usage: iosim [-N nfiles] [-n numwriters] [-s items] [-w width] [-p path] [-m (create|update|read)] [-d (to delete fakedata afterwards)] filename\n");
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

typedef struct ptrlog{
    double * create;
    double * open;
    double * write;
    double * read;
    double * close;
} ptrlog;
typedef struct log{
    double create;
    double open;
    double write;
    double read;
    double close;
} log;

static void
sim(int Nfile, int Nwriter, size_t Nitems, char * filename, double * tlog_ranks, ptrlog tlog, log * times, int mode)
{
    info("Writing to `%s`\n", filename);
    info("Physical Files %d\n", Nfile);
    info("Ranks %d\n", NTask);
    info("Writers %d\n", Nwriter);
    info("Bytes Per Rank %td\n", Nitems * 4 / NTask);
    info("Items Per Rank %td\n", Nitems / NTask);

    size_t itemsperrank = 1024;
    itemsperrank = Nitems / NTask;
    
    BigFile bf = {0};
    BigBlock bb = {0};
    BigArray array = {0};
    BigBlockPtr ptr = {0};

    int * fakedata;
    ptrdiff_t i;
    
    //+++++++++++++++++ Timelog variables +++++++++++++++++
    double t0, t1;
    log trank;
    trank.create = trank.open = trank.write = trank.read = trank.close = 0;
    int nel_trank = sizeof(trank) / sizeof(trank.create);
/*    MPI_Datatype MPI_TIMELOG;
    MPI_Aint displacement[nel_trank], dblex;
    MPI_Type_extent(MPI_DOUBLE, &dblex);
    displacement[0] = (MPI_Aint)0;
    for (i=1; i < nel_trank; i ++) displacement[i] = i*dblex;
    MPI_Type_struct( 1, nel_trank, displacement, MPI_DOUBLE, &MPI_TIMELOG);
    MPI_Type_commit( &MPI_TIMELOG);
*/
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
        big_file_mpi_create_block(&bf, &bb, "TestBlock", "i4", 1, Nfile, Nitems, MPI_COMM_WORLD);
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
        if(bb.size != Nitems) {
            info("Size mismatched, overriding Nitems = %td\n", bb.size);
            Nitems = bb.size;
            itemsperrank = Nitems / NTask;
        }
        if(bb.Nfile != Nfile) {
            info("Nfile mismatched, overriding Nfile = %d\n", bb.Nfile );
            Nfile = bb.Nfile;
        }
    }

    fakedata = malloc(4 * itemsperrank);
    big_array_init(&array, fakedata, "i4", 1, &itemsperrank, NULL);

    if(mode == MODE_CREATE || mode == MODE_UPDATE) {
        info("Initializing FakeData\n");
        for(i = 0; i < itemsperrank; i ++) {
            fakedata[i] = itemsperrank * ThisTask + i;
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
        for(i = 0; i < itemsperrank; i ++) {
            if(fakedata[i] != itemsperrank * ThisTask + i) {
                info("data is corrupted either due to reading or writing\n");
                abort();
            }
        }
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

    //+++++++++++++++++ Preparing Time Log using MPI_Send & MPI_Recv +++++++++++++++++
    /*        if(ThisTask != 0) {
     MPI_Send(&trank.write, 1, MPI_DOUBLE, 0, ThisTask, MPI_COMM_WORLD);
     } else if (ThisTask == 0) {
     tlog_ranks[ThisTask] = trank.write;
     for (i=1; i<NTask; i++) {
     MPI_Status status;
     MPI_Recv(&trank.write, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
     tlog_ranks[status.MPI_SOURCE] = trank.write;
     }
     }
     */
    //+++++++++++++++++ Now using MPI_Gather +++++++++++++++++
/*    MPI_Gather(&trank.create, 1, MPI_DOUBLE, tlog.create, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&trank.open, 1, MPI_DOUBLE, tlog.open, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&trank.write, 1, MPI_DOUBLE, tlog.write, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&trank.read, 1, MPI_DOUBLE, tlog.read, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&trank.close, 1, MPI_DOUBLE, tlog.close, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
*/
    MPI_Gather(&trank, nel_trank, MPI_DOUBLE, times, nel_trank, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //MPI_Gather(&trank, 1, MPI_TIMELOG, times, 1, MPI_TIMELOG, 0, MPI_COMM_WORLD);
    //+++++++++++++++++ END +++++++++++++++++
    free(fakedata);
}

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    
    int i;
    int ch;
    int width = 1;
    int Nfile = 1;
    int Nwriter = NTask;
    size_t Nitems = 1024;
    char * filename = alloca(1500);
    char * path = "";
    char * postfix = alloca(1500);
    int mode = MODE_CREATE;
    int delfiles = 0;
    char * buffer = alloca(1500);
    //+++++++++++++++++ Timelog +++++++++++++++++
    char * timelog = alloca(1000);
    double * tlog_ranks = (double *) malloc(sizeof(double)*NTask);
    FILE *F;
    ptrlog tlog;
    tlog.create = (double *) malloc(sizeof(double)*NTask);
    tlog.open = (double *) malloc(sizeof(double)*NTask);
    tlog.write = (double *) malloc(sizeof(double)*NTask);
    tlog.read = (double *) malloc(sizeof(double)*NTask);
    tlog.close = (double *) malloc(sizeof(double)*NTask);
    log * times = malloc(sizeof(log) * NTask);
    
    while(-1 != (ch = getopt(argc, argv, "hN:n:s:w:p:m:d"))) {
        switch(ch) {
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
                if(1 != sscanf(optarg, "%d", &width)) {
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
                if(1 != sscanf(optarg, "%td", &Nitems)) {
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
    Nitems *= width;
    if (Nitems % NTask != 0) {
        Nitems -= Nitems % NTask;
        info("#items not divisible by ranks!\n Overriding total#items = %td\n", Nitems);
    }
    if (Nwriter > NTask) {
        info("\n\n ############## CAUTION: you chose %d ranks and %d writers! ##############\n"
             " #  If you want %d writers, allocate at least %d ranks with <mpirun -n %d> #\n"
             " ################### Can only use %d writers instead! ###################\n\n",
             NTask, Nwriter, Nwriter, Nwriter, Nwriter, NTask);
        Nwriter = NTask;
    }

//+++++++++++++++++ Starting Simulation +++++++++++++++++
    sim(Nfile, Nwriter, Nitems, filename, tlog_ranks, tlog, times, mode);
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
            fprintf(F, "# files %d ranks %d writers %d items %td width %d\n",
                         Nfile, NTask, Nwriter, Nitems, width);
            //fprintf(F, "Task\tTwrite\t\tTlog.write\n");
            fprintf(F, "# Task\tTcreate\t\tTopen\t\tTwrite\t\tTread\t\tTclose\n");
            for (i=0; i<NTask; i++) {
                //fprintf(F, "%d\t%f\t%f\n", i, tlog_ranks[i], tlog.write[i]);
                //fprintf(F, "%d\t%f\t%f\t%f\t%f\t%f\n", i, tlog.create[i], tlog.open[i], tlog.write[i], tlog.read[i], tlog.close[i]);
                fprintf(F, "%d\t%f\t%f\t%f\t%f\t%f\n", i, times[i].create, times[i].open, times[i].write, times[i].read, times[i].close);
            }
        }
        fclose(F);
    }

byebye:
    free(tlog_ranks);
    free(tlog.create);free(tlog.open);free(tlog.write);free(tlog.read);free(tlog.close);
    free(times);
    MPI_Finalize();
    return 0;
}
