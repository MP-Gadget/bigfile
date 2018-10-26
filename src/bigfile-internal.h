
int _big_file_mksubdir_r(const char * pathname, const char * subdir);

int _dtype_convert(BigArrayIter * dst, BigArrayIter * src, size_t nmemb);

/* The internal code creates the meta data but not the physical back-end storage files */
int _big_block_create_internal(BigBlock * bb, const char * basename, const char * dtype, int nmemb, int Nfile, const size_t fsize[]);

/* the internal code to free meta-data associated with a block. */
void _big_block_close_internal(BigBlock * block);

/* The internal code grows the internal meta data but not the physical back-end stroage files */
int _big_block_grow_internal(BigBlock * bb, int Nfile_grow, const size_t fsize_grow[]);

/* The internal routine to open a physical file */
FILE * _big_file_open_a_file(const char * basename, int fileid, char * mode, int raise);

/* Internal routine to serialize/deserialize a attrset */
void * _big_attrset_pack(BigAttrSet * attrset, size_t * bytes);
BigAttrSet * _big_attrset_unpack(void * p);

int _dtype_normalize(char * dst, const char * src);

