#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <mpi.h>
#include "mp-mpiu.h"

static int
_MPIU_Segmenter_assign_segment_numbers(size_t glocalsize, size_t * sizes, int * nsegment, MPI_Comm comm)
{
    int NTask;
    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);

    int i;
    /* no data for color of -1; exclude them later with special cases */
    int mysegment = -1;
    size_t current_size = 0;
    int current_segment = 0;
    for(i = 0; i < NTask; i ++) {
        current_size += sizes[i];
        /* Assign a colour to this task,
         * maximally equal to the number of
         * tasks with data before this one.*/
        if(i == ThisTask && sizes[i] > 0) {
            mysegment = current_segment;
        }

        /* Start a new segment if we have too much data
         * for this one and more tasks to assign*/
        if(current_size > glocalsize && i < NTask -1) {
            current_size = 0;
            current_segment ++;
        }
    }
    *nsegment = current_segment + 1;
    return mysegment;
}

void
MPIU_Segmenter_init(MPIU_Segmenter * segmenter,
               size_t * sizes,
               size_t totalsize,
               size_t maxsegsize,
               size_t minsegsize,
               int Ngroup,
               MPI_Comm comm)
{
    int ThisTask, NTask;

    MPI_Comm_size(comm, &NTask);
    MPI_Comm_rank(comm, &ThisTask);

    if(Ngroup <= 0 || Ngroup > NTask)
        Ngroup = NTask;

    /* try to create as many segments as number of groups (thus one segment per group) */
    size_t avgsegsize = totalsize / Ngroup;

    /* For small segments it is more efficient to gather all the data to a single rank in the group and write it at once*/
    if(avgsegsize < minsegsize)
        avgsegsize = minsegsize;

    /* no segment shall exceed the memory bound set by maxsegsize, since it will be collected to a single rank */
    if(avgsegsize > maxsegsize)
        avgsegsize = maxsegsize;

    /* If avgsegsize == 0, this assigns a segment number in order to every rank which has non-zero data.
       If avgsegsize > 0, a new segment number is assigned every time a rank exceeds avgsegsize. */
    segmenter->ThisSegment = _MPIU_Segmenter_assign_segment_numbers(avgsegsize, sizes, &segmenter->Nsegments, comm);

    if(segmenter->ThisSegment >= 0) {
        /* assign segments to groups.
         * if Nsegments < Ngroup, some groups will have no segments, and thus no ranks belong to them. */
        segmenter->GroupID = ((size_t) segmenter->ThisSegment) * Ngroup / segmenter->Nsegments;
    } else {
        /* Ranks with no data end up here and in a special group with nothing to do*/
        segmenter->GroupID = Ngroup + 1;
        segmenter->ThisSegment = NTask + 1;
    }

    segmenter->Ngroup = Ngroup;

    /* Split the communicator into groups of segments*/
    MPI_Comm_split(comm, segmenter->GroupID, ThisTask, &segmenter->Group);

    /* Find the minimum and maximum rank of this segment in this group*/
    MPI_Allreduce(&segmenter->ThisSegment, &segmenter->segment_start, 1, MPI_INT, MPI_MIN, segmenter->Group);
    MPI_Allreduce(&segmenter->ThisSegment, &segmenter->segment_end, 1, MPI_INT, MPI_MAX, segmenter->Group);

    segmenter->segment_end ++;

    MPI_Comm_split(segmenter->Group, segmenter->ThisSegment, ThisTask, &segmenter->Segment);

    /* rank with least data in a segment is the leader of the segment.
     * Use the rank within Segment (not the global rank) as the identifier. */
    int segment_rank;
    MPI_Comm_rank(segmenter->Segment, &segment_rank);
    struct { long val; int rank; } local = { (long)sizes[ThisTask], segment_rank };
    struct { long val; int rank; } result;
    MPI_Allreduce(&local, &result, 1, MPI_LONG_INT, MPI_MINLOC, segmenter->Segment);
    segmenter->segment_leader_rank = result.rank;
}

void
MPIU_Segmenter_destroy(MPIU_Segmenter * segmenter)
{
    MPI_Comm_free(&segmenter->Segment);
    MPI_Comm_free(&segmenter->Group);
}

