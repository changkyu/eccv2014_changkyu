/*
 * mvt_param.h
 *
 *  Created on: Jul 7, 2013
 *      Author: changkyu
 */

#ifndef MVT_PARAM_H_
#define MVT_PARAM_H_

#define MVT_PARAM__HELP               ("help")
#define MVT_PARAM__IMG_PATH           ("img_path")
#define MVT_PARAM__IMG_LIST           ("img_list")
#define MVT_PARAM__IMG_NAMEFORMAT     ("img_nameformat")
#define MVT_PARAM__IMG_START_FRAME    ("img_start_frame")
#define MVT_PARAM__IMG_END_FRAME      ("img_end_frame")
#define MVT_PARAM__DPMCONF_PATH       ("dpmconf_path")
#define MVT_PARAM__DPMCONF_NAMEFORMAT ("dpmconf_nameformat")
#define MVT_PARAM__RESULT_FILE        ("result_file")
#define MVT_PARAM__LOG_FILE           ("log_file")
#define MVT_PARAM__DPM_ONOFF          ("dpm_ONOFF")
#define MVT_PARAM__ALM_ONOFF          ("alm_ONOFF")
#define MVT_PARAM__MIL_ONOFF          ("mil_ONOFF")
#define MVT_PARAM__PAIRWISE_ONOFF     ("pairwise_ONOFF")
#define MVT_PARAM__PRIOR_ONOFF        ("prior_ONOFF")
#define MVT_PARAM__USE_MIL_ROOT       ("use_mil_root")
#define MVT_PARAM__VIS                ("vis")

#define MVT_PARAM__INIT_ONOFF     ("init_ONOFF")
#define MVT_PARAM__INIT_STATE_X	   ("init_state_x")
#define MVT_PARAM__INIT_STATE_Y	   ("init_state_y")
#define MVT_PARAM__INIT_STATE_A	   ("init_state_a")
#define MVT_PARAM__INIT_STATE_E	   ("init_state_e")
#define MVT_PARAM__INIT_STATE_D	   ("init_state_d")

#define MVT_PARAM__OBJECT_CATEGORY    ("object_category")
#define MVT_PARAM__3DOBJECT_PATH      ("3dobject_path")
#define MVT_PARAM__NUM_OF_VIEWPOINT   ("num_of_viewpoint_sample")
#define MVT_PARAM__NUM_OF_CENTER      ("num_of_center_sample")
#define MVT_PARAM__NUM_OF_PARTCENTER  ("num_of_partcenter_sample")
#define MVT_PARAM__STD_AZIMUTH        ("std_azimuth")
#define MVT_PARAM__STD_ELEVATION      ("std_elevation")
#define MVT_PARAM__STD_DISTANCE       ("std_distance")
#define MVT_PARAM__STD_PRIOR_AZIMUTH  ("std_prior_azimuth")
#define MVT_PARAM__STD_PRIOR_ELEVATION ("std_prior_elevation")
#define MVT_PARAM__STD_PRIOR_DISTANCE  ("std_prior_distance")
#define MVT_PARAM__SEARCH_WINDOW_SIZE ("search_window_size")

#define MVT_PARAM__THRESH_DPM         ("thresh_dpm")
#define MVT_PARAM__THRESH2_DPM         ("thresh2_dpm")
#define MVT_PARAM__THRESH_ALM         ("thresh_alm")
#define MVT_PARAM__THRESH_MIL         ("thresh_mil")

#define MVT_PARAM__WEIGHT_MIL_ROOT    ("weight_mil_root")
#define MVT_PARAM__HEIGHT_MIL_ROOT     ("height_mil_root")

MVT_Param ParseArguments(int argc, char **argv);
void PrintArguments();

extern MVT_Param g_param;

#endif /* MVT_PARAM_H_ */
