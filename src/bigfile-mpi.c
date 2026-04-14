#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <alloca.h>
#include <string.h>
#include "bigfile-mpi.h"
#include "bigfile-internal.h"
#include "mp-mpiu.h"

/* disable aggregation by default */
static size_t _BigFileAggThreshold = 0;

static int big_block_mpi_broadcast(BigBlock * bb, int root, MPI_Comm comm);
static int big_file_mpi_broadcast_anyerror(int rt, MPI_Comm comm);

#define BCAST_AND_RAISEIF(rt, comm) \
    if(0 != (rt = big_file_mpi_broadcast_anyerror(rt, comm))) { \
        return rt; \
    } \

void
big_file_mpi_set_verbose(int verbose)
{
}

void
big_file_mpi_set_aggregated_threshold(size_t bytes)
{
    _BigFileAggThreshold = bytes;
}

size_t
big_file_mpi_get_aggregated_threshold()
{
    return _BigFileAggThreshold;
}

int big_file_mpi_open(BigFile * bf, const char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt = 0;
    if (rank == 0) {
        rt = big_file_open(bf, basename);
    } else {
        /* FIXME : */
        bf->basename = _strdup(basename);
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

int big_file_mpi_create(BigFile * bf, const char * basename, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt;
    if (rank == 0) {
        rt = big_file_create(bf, basename);
    } else {
        /* FIXME : */
        bf->basename = _strdup(basename);
        rt = 0;
    }
    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

/**Helper function for big_file_mpi_create_block, above*/
static int
_big_block_mpi_create(BigBlock * bb,
        const char * basename,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm);

/** Helper function for big_file_mpi_open_block, above*/
static int _big_block_mpi_open(BigBlock * bb, const char * basename, MPI_Comm comm);

int big_file_mpi_open_block(BigFile * bf, BigBlock * block, const char * blockname, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    if(!bf || !bf->basename || !blockname) return 1;
    char * basename = (char *) alloca(strlen(bf->basename) + strlen(blockname) + 128);
    sprintf(basename, "%s/%s/", bf->basename, blockname);
    return _big_block_mpi_open(block, basename, comm);
}

int
big_file_mpi_create_block(BigFile * bf,
        BigBlock * block,
        const char * blockname,
        const char * dtype,
        int nmemb,
        int Nfile,
        size_t size,
        MPI_Comm comm)
{
    size_t fsize[Nfile];
    int i;
    for(i = 0; i < Nfile; i ++) {
        fsize[i] = size * (i + 1) / Nfile
                 - size * (i) / Nfile;
    }
    return _big_file_mpi_create_block(bf, block, blockname, dtype,
        nmemb, Nfile, fsize, comm);
}

int
_big_file_mpi_create_block(BigFile * bf,
        BigBlock * block,
        const char * blockname,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm)
{
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);

    int rt = 0;
    if (rank == 0) {
        rt = _big_file_mksubdir_r(bf->basename, blockname);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    char * basename = (char *) alloca(strlen(bf->basename) + strlen(blockname) + 128);
    sprintf(basename, "%s/%s/", bf->basename, blockname);
    return _big_block_mpi_create(block, basename, dtype, nmemb, Nfile, fsize, comm);
}

int big_file_mpi_close(BigFile * bf, MPI_Comm comm) {
    if(comm == MPI_COMM_NULL) return 0;
    int rt = big_file_close(bf);
    return rt;
}

static int
_big_block_mpi_open(BigBlock * bb, const char * basename, MPI_Comm comm)
{
    if(comm == MPI_COMM_NULL) return 0;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int rt;
    if(rank == 0) {
        rt = _big_block_open(bb, basename);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    big_block_mpi_broadcast(bb, 0, comm);
    return 0;
}

static int
_big_block_mpi_create(BigBlock * bb,
        const char * basename,
        const char * dtype,
        int nmemb,
        int Nfile,
        const size_t fsize[],
        MPI_Comm comm)
{
    int rank;
    int NTask;
    int rt;

    if(comm == MPI_COMM_NULL) return 0;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &rank);

    if(rank == 0) {
        rt = _big_block_create_internal(bb, basename, dtype, nmemb, Nfile, fsize);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    big_block_mpi_broadcast(bb, 0, comm);

    int i;
    for(i = (size_t) bb->Nfile * rank / NTask; i < (size_t) bb->Nfile * (rank + 1) / NTask; i ++) {
        FILE * fp = _big_file_open_a_file(bb->basename, i, "w", 1);
        if(fp == NULL) {
            rt = -1;
            break;
        }
        fclose(fp);
    }

    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

int
big_block_mpi_grow(BigBlock * bb,
    int Nfile_grow,
    const size_t fsize_grow[],
    MPI_Comm comm) {

    int rank;
    int NTask;
    int rt;

    if(comm == MPI_COMM_NULL) return 0;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &rank);

    int oldNfile = bb->Nfile;

    if(rank == 0) {
        rt = _big_block_grow_internal(bb, Nfile_grow, fsize_grow);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);

    if(rank != 0) {
        /* closed on non-root because we will bcast.*/
        _big_block_close_internal(bb);
    }
    big_block_mpi_broadcast(bb, 0, comm);

    int i;
    for(i = (size_t) Nfile_grow * rank / NTask; i < (size_t) Nfile_grow * (rank + 1) / NTask; i ++) {
        FILE * fp = _big_file_open_a_file(bb->basename, i + oldNfile, "w", 1);
        if(fp == NULL) {
            rt = -1;
            break;
        }
        fclose(fp);
    }

    BCAST_AND_RAISEIF(rt, comm);

    return rt;
}

int
big_block_mpi_grow_simple(BigBlock * bb, int Nfile_grow, size_t size_grow, MPI_Comm comm)
{
    size_t fsize[Nfile_grow];
    int i;
    for(i = 0; i < Nfile_grow; i ++) {
        fsize[i] = size_grow * (i + 1) / Nfile_grow
                 - size_grow * (i) / Nfile_grow;
    }
    int rank;
    MPI_Comm_rank(comm, &rank);

    return big_block_mpi_grow(bb, Nfile_grow, fsize, comm);
}


int
big_block_mpi_flush(BigBlock * block, MPI_Comm comm)
{
    if(comm == MPI_COMM_NULL) return 0;

    int rank;
    MPI_Comm_rank(comm, &rank);

    unsigned int * checksum = (unsigned int *) alloca(sizeof(int) * block->Nfile);
    MPI_Reduce(block->fchecksum, checksum, block->Nfile, MPI_UNSIGNED, MPI_SUM, 0, comm);
    int dirty;
    MPI_Reduce(&block->dirty, &dirty, 1, MPI_INT, MPI_LOR, 0, comm);
    int rt;
    if(rank == 0) {
        /* only the root rank updates */
        int i;
        big_block_set_dirty(block, dirty);
        for(i = 0; i < block->Nfile; i ++) {
            block->fchecksum[i] = checksum[i];
        }
        rt = big_block_flush(block);
    } else {
        rt = 0;
    }

    BCAST_AND_RAISEIF(rt, comm);
    /* close as we will broadcast the block */
    if(rank != 0) {
        _big_block_close_internal(block);
    }
    big_block_mpi_broadcast(block, 0, comm);
    return 0;

}
int big_block_mpi_close(BigBlock * block, MPI_Comm comm) {

    int rt = big_block_mpi_flush(block, comm);
    _big_block_close_internal(block);

    return rt;
}

static int
big_file_mpi_broadcast_anyerror(int rt, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int root, loc = 0;
    /* Add 1 so we do not swallow errors on rank 0*/
    if(rt != 0)
        loc = rank+1;

    MPI_Allreduce(&loc, &root, 1, MPI_INT, MPI_MAX, comm);

    if (root == 0) {
        /* no errors */
        return 0;
    }

    root -= 1;
    char * error = big_file_get_error_message();

    int errorlen;
    if(rank == root) {
        errorlen = strlen(error);
    }
    MPI_Bcast(&errorlen, 1, MPI_INT, root, comm);

    if(rank != root) {
        error = (char *) malloc(errorlen + 1);
    }

    MPI_Bcast(error, errorlen + 1, MPI_BYTE, root, comm);

    if(rank != root) {
        big_file_set_error_message(error);
        free(error);
    }

    MPI_Bcast(&rt, 1, MPI_INT, root, comm);

    return rt;
}

static int
big_block_mpi_broadcast(BigBlock * bb, int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    void * buf = NULL;
    size_t bytes = 0;

    if(rank == root) {
        buf = _big_block_pack(bb, &bytes);
    }

    MPI_Bcast(&bytes, sizeof(bytes), MPI_BYTE, root, comm);

    if(rank != root) {
        buf = malloc(bytes);
    }

    MPI_Bcast(buf, bytes, MPI_BYTE, root, comm);

    if(rank != root) {
        _big_block_unpack(bb, buf);
    }
    free(buf);
    return 0;
}

static int
_aggregated(
            BigBlock * block,
            BigBlockPtr * ptr,
            ptrdiff_t offset, /* offset of the entire comm */
            size_t localsize,
            BigArray * array,
            int write,
            int root,
            const char * mode,
            MPI_Comm comm);

static int
_throttle_action(MPI_Comm comm, int concurrency, BigBlock * block,
    BigBlockPtr * ptr,
    BigArray * array,
    int write)
{
    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    MPIU_Segmenter seggrp[1];

    size_t totalsize = 0;
    size_t localsize = array->dims[0];
    size_t myoffset = 0;
    size_t * sizes = (size_t *) malloc(sizeof(sizes[0]) * NTask);
    sizes[ThisTask] = localsize;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sizes, 1, MPI_UNSIGNED_LONG, comm);
    int i;
    for(i = 0; i < ThisTask; i ++)
        myoffset += sizes[i];
    for(i = 0; i < NTask; i ++)
        totalsize += sizes[i];


    size_t minsegsize = 32 * 1024 * 1024;
    /* Creates segments and groups. The number of groups is roughly equal
     * to the number of writing processes (with a complexity if some processes have no data to write).
     * The number of segments is set by the average size of data to write to a file.*/
    MPIU_Segmenter_init(seggrp, sizes, totalsize, _BigFileAggThreshold, minsegsize, concurrency, comm);

    free(sizes);

    int rt = 0;
    int segment;

    for(segment = seggrp->segment_start;
        segment < seggrp->segment_end;
        segment ++) {

        /* Ensures that ranks in this group, but not in this segment do not try to write at the same time*/
        MPI_Barrier(seggrp->Group);

        /* This extra broadcast is so that the error-ing segment stops trying to write.*/
        if(0 != (rt = big_file_mpi_broadcast_anyerror(rt, seggrp->Group))) {
            /* failed , no more writes to segment. */
            continue;
        }
        if(seggrp->ThisSegment != segment) continue;

        /* use the offset on the first task in the SegGroup */
        size_t offset = myoffset;
        MPI_Bcast(&offset, 1, MPI_UNSIGNED_LONG, 0, seggrp->Segment);

        rt = _aggregated(block, ptr, offset, localsize, array, write, seggrp->segment_leader_rank, "r+", seggrp->Segment);
    }

    if(0 == (rt = big_file_mpi_broadcast_anyerror(rt, comm))) {
        /* no errors*/
        big_block_seek_rel(block, ptr, totalsize);
    }

    MPIU_Segmenter_destroy(seggrp);
    return rt;
}

static int
_aggregated(
            BigBlock * block,
            BigBlockPtr * ptr,
            ptrdiff_t offset, /* offset of the entire comm */
            size_t localsize,
            BigArray * array,
            int write,
            int root,
            const char * mode,
            MPI_Comm comm)
{
    size_t elsize = big_file_dtype_itemsize(block->dtype) * block->nmemb;

    /* This will aggregate to the root and write */
    BigBlockPtr ptr1[1];
    /* use memcpy because older compilers doesn't like *ptr assignments */
    memcpy(ptr1, ptr, sizeof(BigBlockPtr));

    int i;
    int e = 0;
    int rank;
    int nrank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nrank);

    BigArray garray[1], larray[1];
    BigArrayIter iarray[1], ilarray[1];
    void * lbuf = malloc(elsize * localsize);
    void * gbuf = NULL;

    int recvcounts[nrank];
    int recvdispls[nrank + 1];

    recvdispls[0] = 0;
    recvcounts[rank] = localsize;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvcounts, 1, MPI_INT, comm);

    for(i = 0; i < nrank; i ++) {
        recvdispls[i + 1] = recvdispls[i] + recvcounts[i];
    }

    size_t grouptotalsize = recvdispls[nrank];

    MPI_Datatype mpidtype;
    MPI_Type_contiguous(elsize, MPI_BYTE, &mpidtype);
    MPI_Type_commit(&mpidtype);

    big_array_init(larray, lbuf, block->dtype, 2, (size_t[]){localsize, (size_t) block->nmemb}, NULL);

    big_array_iter_init(iarray, array);
    big_array_iter_init(ilarray, larray);

    if(rank == root) {
        gbuf = malloc(grouptotalsize * elsize);
        big_array_init(garray, gbuf, block->dtype, 2, (size_t[]){grouptotalsize, (size_t) block->nmemb}, NULL);
    }

    if(write) {
        _dtype_convert(ilarray, iarray, localsize * block->nmemb);
        MPI_Gatherv(lbuf, recvcounts[rank], mpidtype,
                    gbuf, recvcounts, recvdispls, mpidtype, root, comm);
    }
    if(rank == root) {
        big_block_seek_rel(block, ptr1, offset);
        if(write)
            e = _big_block_write_mode(block, ptr1, garray, mode);
        else
            e = big_block_read(block, ptr1, garray);
    }
    /* We are a read*/
    if(!write) {
        MPI_Scatterv(gbuf, recvcounts, recvdispls, mpidtype,
                    lbuf, localsize, mpidtype, root, comm);
        _dtype_convert(iarray, ilarray, localsize * block->nmemb);
    }

    if(rank == root) {
        free(gbuf);
    }
    free(lbuf);

    MPI_Type_free(&mpidtype);

    return big_file_mpi_broadcast_anyerror(e, comm);
}

int
big_block_mpi_create_and_write(BigFile * bf,
        const char * blockname,
        BigArray * array,
        int concurrency,
        MPI_Comm comm)
{
    int NTask, ThisTask;
    int rt = 0;

    if(comm == MPI_COMM_NULL) return 0;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    if (ThisTask == 0) {
        rt = _big_file_mksubdir_r(bf->basename, blockname);
    }

    BCAST_AND_RAISEIF(rt, comm);

    MPIU_Segmenter seggrp[1];

    size_t totalsize = 0;
    size_t localsize = array->dims[0];
    size_t * sizes = (size_t *) malloc(sizeof(sizes[0]) * NTask);
    sizes[ThisTask] = localsize;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sizes, 1, MPI_UNSIGNED_LONG, comm);
    int i;
    for(i = 0; i < NTask; i ++)
        totalsize += sizes[i];

    /* For datasets smaller than some value, it is more efficient
     * to gather them to one rank, rather than opening the file on each rank
     * in the group in turn. The number here is not rigorously chosen, but is smaller
     * than CHUNK_BYTES so we can definitely do it in one write*/
    size_t minsegsize = 32 * 1024 * 1024;

    /* Creates segments and groups. The number of groups is roughly equal
     * to the number of writing processes (with a complexity if some processes have no data to write).
     * We create one file per segment group.*/
    MPIU_Segmenter_init(seggrp, sizes, totalsize, totalsize, minsegsize, concurrency, comm);

    size_t group_total;
    MPI_Allreduce(&sizes[ThisTask], &group_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, seggrp->Group);
    /* Work out the sizes of each group. This will become the sizes for the files*/
    size_t * gsizes = calloc(seggrp->Ngroup, sizeof(size_t));
    if(seggrp->GroupID < seggrp->Ngroup)
        gsizes[seggrp->GroupID] = group_total;
    /* At this point elements in the group have gsizes = group_total for their own group, and
     * zero for the other groups. Propagate the group sizes below, noting we use MPI_MAX */
    MPI_Allreduce(MPI_IN_PLACE, gsizes, seggrp->Ngroup, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    BigBlock block = {0};
    if(ThisTask == 0) {
        /* Now we know the groups and segments, create the files*/
        size_t stringlen = strlen(bf->basename) + strnlen(blockname, 512) + 128;
        char * basename = (char *) malloc(stringlen);
        snprintf(basename, stringlen, "%s/%s/", bf->basename, blockname);
        rt = _big_block_create_internal(&block, basename, array->dtype, array->dims[1], seggrp->Ngroup, gsizes);
        free(basename);
    }

    size_t myoffset = 0;
    for(i = 0; i < ThisTask; i ++)
        myoffset += sizes[i];

    free(gsizes);
    free(sizes);

    if(0 != (rt = big_file_mpi_broadcast_anyerror(rt, comm))) {
        /* No need to close the block as if rt != 0 we didn't open it correctly*/
        MPIU_Segmenter_destroy(seggrp);
        return rt;
    }

    big_block_mpi_broadcast(&block, 0, comm);

    BigBlockPtr ptr = {0};

    int segment;

    for(segment = seggrp->segment_start;
        segment < seggrp->segment_end;
        segment ++) {

        /* Ensures that ranks in this group, but not in this segment do not try to write at the same time*/
        MPI_Barrier(seggrp->Group);

        /* This extra broadcast is so that the error-ing segment stops trying to write.*/
        if(0 != (rt = big_file_mpi_broadcast_anyerror(rt, seggrp->Group))) {
            /* failed , no more writes to segment. */
            continue;
        }
        if(seggrp->ThisSegment != segment) continue;

        /* use the offset on the first task in the SegGroup */
        size_t offset = myoffset;
        MPI_Bcast(&offset, 1, MPI_UNSIGNED_LONG, 0, seggrp->Segment);

        /* write = 1 : Always writing here and we use mode 'w' so we create the files.*/
        rt = _aggregated(&block, &ptr, offset, localsize, array, 1, seggrp->segment_leader_rank, "w", seggrp->Segment);
    }

    MPIU_Segmenter_destroy(seggrp);

    /* Block written, close it: we need to close even if we have a write error,
     * but we want to preserve the write error in that case.*/
    int close_rt = big_block_mpi_close(&block, comm);
    if(rt == 0)
        rt = close_rt;
    rt = big_file_mpi_broadcast_anyerror(rt, comm);
    return rt;
}


int
big_block_mpi_write(BigBlock * block, BigBlockPtr * ptr, BigArray * array, int concurrency, MPI_Comm comm)
{
    int rt = _throttle_action(comm, concurrency, block, ptr, array, 1);
    return rt;
}

int
big_block_mpi_read(BigBlock * block, BigBlockPtr * ptr, BigArray * array, int concurrency, MPI_Comm comm)
{
    int rt = _throttle_action(comm, concurrency, block, ptr, array, 0);
    return rt;
}


int
big_file_mpi_create_records(BigFile * bf,
    const BigRecordType * rtype,
    const char * mode,
    int Nfile,
    const size_t fsize[],
    MPI_Comm comm)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigBlock block[1];
        if (0 == strcmp(mode, "w+")) {
            RAISEIF(0 != _big_file_mpi_create_block(bf, block,
                             rtype->fields[i].name,
                             rtype->fields[i].dtype,
                             rtype->fields[i].nmemb,
                             Nfile,
                             fsize,
                             comm),
                ex_open,
                NULL);
        } else if (0 == strcmp(mode, "a+")) {
            RAISEIF(0 != big_file_mpi_open_block(bf, block, rtype->fields[i].name, comm),
                ex_open,
                NULL);
            RAISEIF(0 != big_block_mpi_grow(block, Nfile, fsize, comm),
                ex_grow,
                NULL);
        } else {
            RAISE(ex_open,
                "Mode string must be `a+` or `w+`, `%s` provided",
                mode);
        }
        RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
        continue;
        ex_grow:
            RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
            return -1;
    }
    return 0;
}
int
big_file_mpi_write_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    const void * buf,
    int concurrency,
    MPI_Comm comm)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigArray array[1];
        BigBlock block[1];
        BigBlockPtr ptr = {0};

        /* rainwoodman: cast away the const. We don't really modify it.*/
        RAISEIF(0 != big_record_view_field(rtype, i, array, size, (void*) buf),
            ex_array,
            NULL);
        RAISEIF(0 != big_file_mpi_open_block(bf, block, rtype->fields[i].name, comm),
            ex_open,
            NULL);
        RAISEIF(0 != big_block_seek(block, &ptr, offset),
            ex_seek,
            NULL);
        RAISEIF(0 != big_block_mpi_write(block, &ptr, array, concurrency, comm),
            ex_write,
            NULL);
        RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
        continue;
        ex_write:
        ex_seek:
            RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
        ex_array:
            return -1;
    }
    return 0;
}


int
big_file_mpi_read_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    void * buf,
    int concurrency,
    MPI_Comm comm)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigArray array[1];
        BigBlock block[1];
        BigBlockPtr ptr = {0};

        RAISEIF(0 != big_record_view_field(rtype, i, array, size, buf),
            ex_array,
            NULL);
        RAISEIF(0 != big_file_mpi_open_block(bf, block, rtype->fields[i].name, comm),
            ex_open,
            NULL);
        RAISEIF(0 != big_block_seek(block, &ptr, offset),
            ex_seek,
            NULL);
        RAISEIF(0 != big_block_mpi_read(block, &ptr, array, concurrency, comm),
            ex_read,
            NULL);
        RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
        continue;
        ex_read:
        ex_seek:
            RAISEIF(0 != big_block_mpi_close(block, comm),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
        ex_array:
            return -1;
    }
    return 0;
}
