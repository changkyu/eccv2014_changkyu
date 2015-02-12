/***********************************************************************/
/*                                                                     */
/*   svm_struct_api_types.h                                            */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 13.10.03                                                    */
/*   Modified by: Yu Xiang                                             */
/*   Date: 05.01.12                                                    */
/*                                                                     */
/*   Copyright (c) 2003  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#ifndef svm_struct_api_types
#define svm_struct_api_types

# include "svm_light/svm_common.h"
# include "svm_light/svm_learn.h"
# include "matrix.h"
# include "cad.h"

# define INST_NAME          "Generic and empty API"
# define INST_VERSION       "V0.00"
# define INST_VERSION_DATE  "??.??.??"

/* default precision for solving the optimization problem */
# define DEFAULT_EPS         0.1 
/* default loss rescaling method: 1=slack_rescaling, 2=margin_rescaling */
# define DEFAULT_RESCALING   2
/* default loss function: */
# define DEFAULT_LOSS_FCT    0
/* default optimization algorithm to use: */
# define DEFAULT_ALG_TYPE    3
/* store Psi(x,y) (for ALG_TYPE 1) instead of recomputing it every time: */
# define USE_FYCACHE         1
# define MINUS_INFINITY -1.0E32
# define PLUS_INFINITY 1.0E32
#define MAX_LABEL_NUM 5000
#define BUFFLE_SIZE 10000000

typedef struct constset { /* a set of linear inequality constrains of
			     for lhs[i]*w >= rhs[i] */
  int     m;            /* m is the total number of constrains */
  DOC     **lhs;
  double  *rhs;
} CONSTSET;

typedef struct pattern {
  /* this defines the x-part of a training example, e.g. the structure
     for storing a natural language sentence in NLP parsing */
  CUMATRIX image;
} PATTERN;

typedef struct label {
  /* this defines the y-part (the label) of a training example,
     e.g. the parse tree of the corresponding sentence. */
  /* object_label = +1: object, object_label = -1: non-object */
  int object_label;
  /* part location in the image, part_num*2 matrix */
  int part_num;
  float *part_label;
  /* occlusion label */
  int *occlusion;
  /* cad model index */
  int cad_label;
  /* viewpoint label */
  int view_label;
  /* bounding box */
  float bbox[4];
  float energy;
} LABEL;

typedef struct energy_index
{
  int index;
  double energy;
} ENERGYINDEX;

typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  /* float weights */
  float *weights;
  /* cad models */
  int cad_num;
  CAD **cads;
  /* log file */
  FILE *fp;
} STRUCTMODEL;

typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  double newconstretrain;      /* number of new constraints to
				  accumulate before recomputing the QP
				  solution (used in w=1 algorithm) */
  int    ccache_size;          /* maximum number of constraints to
				  cache for each example (used in w=4
				  algorithm) */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][300]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss type from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* further parameters that are passed to init_struct_model() */
  char confile[200];           /* file for constraints */
  char cls[200];                /* class name */
  double object_loss;
  double cad_loss;
  double view_loss;
  double location_loss;
  double loss_value;
  double wpair;
  int hard_negative;
  int iter;                  /* training iteration number */
  int is_root;
  int is_aspectlet;
  /* padding */
  int padx;
  int pady;
  int cad_index;
  int part_index;
  int deep;
  /* support vectors */
  CONSTSET cset;
} STRUCT_LEARN_PARM;

typedef struct struct_test_stats {
  /* you can add variables for keeping statistics when evaluating the
     test predictions in svm_struct_classify. This can be used in the
     function eval_prediction and print_struct_testing_stats. */
} STRUCT_TEST_STATS;

#endif
