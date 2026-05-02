#ifndef _MPIU_H_
#define _MPIU_H_

/* Segment a MPI Comm into 'groups', such that distributed data in each group is roughly even.
 * NOTE: this API needs some revision to incorporate some of the downstream behaviors. Currently
 * the internal data structure is directly accessed by downstream.
 * */
typedef struct MPIU_Segmenter {
    /* data model: rank <- segment <- group */
    int Ngroup;
    int Nsegments;
    int GroupID; /* ID of the group of this rank */
    int ThisSegment; /* SegmentID of the local data chunk on this rank*/

    int segment_start; /* segments responsible in this group */
    int segment_end;

    int segment_leader_rank;
    MPI_Comm Group;  /* communicator for all ranks in the group */
    MPI_Comm Segment; /* communicator for all ranks in this segment */
} MPIU_Segmenter;

/* MPIU_segmenter_init: Create a Segmenter.
 * the total number of items according to both sizes and sizes2 will not
 * exceed the epxected_segsize by too much.
 * */
void
MPIU_Segmenter_init(MPIU_Segmenter * segmenter,
               size_t * sizes,   /* IN: size per rank, used to bound the number of ranks in a group. */
               size_t totalsize, /* Total size of data to segment into groups. */
               size_t maxsegsize, /* Maximum desired segment size. Can still be exceeded if a single rank has more data than this.*/
               size_t minsegsize, /* Minimum desired segment size. Segments smaller than this are gathered to one rank*/
               int Ngroup,  /* number of groups to form. */
               MPI_Comm comm);
void
MPIU_Segmenter_destroy(MPIU_Segmenter * segmenter);

#endif
