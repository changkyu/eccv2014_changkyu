/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.h                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*   Modified by: Yu Xiang                                             */
/*   Date: 05.01.12                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include "svm_struct_api_types.h"
#include "svm_struct_common.h"

#ifndef svm_struct_api
#define svm_struct_api

void        svm_struct_learn_api_init(int argc, char* argv[]);
void        svm_struct_learn_api_exit(void);
void        svm_struct_classify_api_init(int argc, char* argv[]);
void        svm_struct_classify_api_exit(void);
SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm, STRUCTMODEL *sm);
void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm);
CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm);
LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm);
LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm);
LABEL* classify_struct_example(PATTERN x, int *label_num, int flag, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
int         empty_label(LABEL y);
SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm, 
	        STRUCT_LEARN_PARM *sparm);
double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);
int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm);
void        print_label(LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm);
void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm,
				       STRUCT_TEST_STATS *teststats);
void        eval_prediction(long exnum, EXAMPLE ex, LABEL prediction, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
			    STRUCT_TEST_STATS *teststats);
void        write_struct_model(char *file,STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm);
STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm);
void        write_label(FILE *fp, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void        free_pattern(PATTERN x);
void        free_label(LABEL y);
void        free_struct_model(STRUCTMODEL sm);
void        free_struct_sample(SAMPLE s);
void        print_struct_help(void);
void        parse_struct_parameters(STRUCT_LEARN_PARM *sparm);
void        print_struct_help_classify(void);
void        parse_struct_parameters_classify(char *attribute, char *value);
CAD** read_cad_model(char *file, int *cad_num_return, int istest, STRUCT_LEARN_PARM *sparm);
CUMATRIX crop_hog(CUMATRIX hog, int cx, int cy, int b0, int b1);
CUMATRIX rectify_potential(CUMATRIX A, int width, int height, int sbin, float *T);
void label_from_backtrack(TREENODE *node, TREENODE *parent, int px, int py, int sbin, int part_num, float *part_dst);
void child_to_parent(TREENODE *node, TREENODE *parent, OBJECT2D *object2d, int **graph, int sbin, float *part_label, STRUCT_LEARN_PARM *sparm);
void copy_to_float_weights(STRUCTMODEL *sm);
void write_weights(STRUCTMODEL *sm);
float compute_azimuth_difference(LABEL y, LABEL ybar, STRUCTMODEL *sm);
void write_constraints(CONSTSET cset, STRUCT_LEARN_PARM *sparm);
void compute_bbox(LABEL *y, STRUCTMODEL *sm);
void compute_root_scores(TREENODE *node, float occ_energy, int sbin, int part_num, float *part_label, STRUCT_LEARN_PARM *sparm);
int* non_maxima_suppression(int *dims, float *data);
void get_multiple_detection(LABEL **ylabel, int *num, int o, int v, TREENODE *node, int *mask, int sbin, int part_num, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void get_multiple_detection_part(LABEL **ylabel, int *num, int o, int v, int *dims, float *potential, int *mask, int sbin, int part_num, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
int compare_label(const void *a, const void *b);
int nms_bbox(LABEL *y, int num);
CONSTSET    read_constraints(char *file, STRUCTMODEL *sm);
void copy_file(char *dst_name, char *src_name);
void combine_files(char *dst_name, char *src_name, int num);
EXAMPLE read_one_example(FILE *fp, CAD **cads);
void compute_bbox_root(LABEL *y, int root_index, CAD *cad);
float get_maximum_label(TREENODE *node, int sbin, int part_num, float *part_label);

#endif
