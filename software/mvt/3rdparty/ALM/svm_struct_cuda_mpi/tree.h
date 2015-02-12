/* A tree data structure for belief propagation algrithm
   Author: Yu Xiang
   Date: 03/23/2011
*/

#ifndef TREE_H
#define TREE_H

typedef struct treenode
{
  int index;

  int dims[2];
  float *potential;

  int message_num;
  float **messages;
  float **locations;

  /* children */
  int child_num;
  struct treenode **children;

  /* parent */
  int parent_num;
  struct treenode **parent;
}TREENODE;

TREENODE* construct_tree(int node_num, int root_index, int **graph);
void construct(TREENODE *node, TREENODE *parent, TREENODE *nodes, int node_num, int **graph);
void print_tree(TREENODE *node);
void initialize_message(TREENODE *nodes, int node_num);
void free_tree(TREENODE *nodes, int node_num);
void set_potential(TREENODE *node, int index, float *potential, int *dims);
void free_message(TREENODE *nodes, int node_num);
int isin_tree(TREENODE *node, int index);

#endif
