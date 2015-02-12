/* CAD model for an object category
   Author: Yu Xiang
   Date: 04/15/2011
*/
#ifndef CAD_H
#define CAD_H

#define HOGLENGTH 32
#define HOGBINSIZE 6

#include "tree.h"

/* data structure for part template */
typedef struct part_template
{
  int width;
  int height;
  /* HOG weights */
  int sbin;
  int b0;
  int b1;
  int length;
  float *weights;
}PARTTEMPLATE;

/* data structure for object in 2D */
typedef struct object2d
{
  /* viewpoint */
  float azimuth;
  float elevation;
  float distance;
  /* viewport size */
  int viewport_size;
  /* number of object parts */
  int part_num;
  /* part location */
  float *part_locations;
  /* flag for occlusion */
  int *occluded;
  /* homographies to transform the 2D part to its frontal view, 9*part_number matrix */
  float **homographies;
  /* part shapes */
  float **part_shapes;
  int** graph;
  int root_index;
  TREENODE *tree;
}OBJECT2D;

typedef struct cad
{
  /* number of object parts */
  int part_num;
  /* root index */
  int *roots;
  /* part names */
  char **part_names;
  /* part template */
  PARTTEMPLATE **part_templates;

  /* number of viewpoints */
  int view_num;
  /* objects in 2D rendered with different viewpoint */
  OBJECT2D **objects2d;
  int feature_len;
}CAD;

CAD* read_cad(FILE *fp, int hog_length);
void destroy_cad(CAD* cad);
void print_cad(CAD *cad);

#endif
