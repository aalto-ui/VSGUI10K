##
library(dplyr)
library(parameters)

###############
## Load data ##
###############

# This is search time data
df <- read.csv("output/data/vsgui10k_search_times.csv", header = T) %>%
  mutate(absent = factor(absent, levels = c(0, 1), labels = c("False", "True")),
         tgt_location = factor(tgt_location, levels = c("upper-left", "upper-right", "lower-right", "lower-left", "between", "absent"), 
                               labels = c("Upper-left", "Upper-right", "Lower-right", "Lower-left", "Between", "Absent")),
         cue = factor(cue, levels = c("i", "t", "tc"), 
                      labels = c("Image", "Text", "Text+color")),
         tgt_color = factor(tgt_color, levels = c("white", "black", "blue", "brown", "green", "grey", "multi", "orange", "pink", "purple", "red", "yellow"), 
                            labels = c("White", "Black", "Blue", "Brown", "Green", "Grey", "Multi", "Orange", "Pink", "Purple", "Red", "Yellow")),
         category = factor(category, levels = c("desktop", "mobile", "web"),
                           labels = c("Desktop", "Mobile", "Web"))) %>%
  ## There seems to be a bad entry, remove it
  filter(search_time > 0)

# This is AIM metrics
results_main <- read.csv("data/vsgui10k_aim_results.csv", header = T)

#################
## AIM METRICS ##
#################

# FA in line with Miniukovich et al. 
# AIM Metrics calculated using https://github.com/aalto-ui/aim

# m1_0	PNG file size in bytes (int, [0, +inf))
# m2_0	JPEG file size in bytes (int, [0, +inf))
# m3_0	Number of distinct RGB values (int, [0, +inf))
# m4_0	Contour density (float, [0, 1])
# m5_0	Figure-ground contrast (float, [0, 1])
# m6_0	Contour congestion (float, [0, 1])
# m7_0	Subband entropy (float, [0, +inf))
# m8_0	Feature congestion (float, [0, +inf))
# m10_0	Average WAVE score across pixels (float, [0, 1))
# m11_0	Number of static color clusters (int, [0, 32^3))
# m12_0	Number of dynamic color clusters (int, [0, +inf))
# m13_0	Luminance standard deviation (float, [0, +inf))
# m14_0	L average (float, [0, 100])
# m14_1	L standard deviation (float, [0, +inf))
# m14_2	A average (float, [-128, +128])
# m14_3	A standard deviation (float, [0, +inf))
# m14_4	B average (float, [-128, +128))
# m14_5	B standard deviation (float, [0, +inf))
# m15_0	Colorfulness (float, [0, +inf))
# m16_0	Hue average (float, [0, +inf))
# m16_1	Saturation average (float, [0, +inf))
# m16_2	Saturation standard deviation (float, [0, +inf))
# m16_3	Value average (float, [0, +inf))
# m16_4	Value standard deviation (float, [0, +inf))
# m17_0	Number of distinct Hue values (int, [1, 255])
# m17_1	Number of distinct Saturation values (int, [1, 255])
# m17_2	Number of distinct Value values (int, [1, 255])
# m18_0	 NIMA mean score (float, [0, 10])
# m18_1	NIMA standard deviation score (float, [0, +inf))
# m19_0	Ratio of distinct RGB values to the number of dynamic clusters (float, [0, +inf))
# m20_0	Distance to the closest harmonic template (float, [0, +inf))
# m21_0	Number of visual GUI blocks (int, [0, +inf))
# m21_1	Number of visual GUI blocks - without children (int, [0, +inf))
# m21_2	Number of alignment points (int, [0, +inf))
# m21_3	 Number of alignment points - without children (int, [0, +inf))
# m21_4	 Number of block sizes (int, [0, +inf))
# m21_5	 Number of block sizes - without children (int, [0, +inf))
# m21_6	GUI coverage (float, [0, 1])
# m21_7	GUI coverage - without children (float, [0, 1])
# m21_8	Number of vertical block sizes (int, [0, +inf))
# m21_9	Number of vertical block sizes - without children (int, [0, +inf))
# m22_0	  White space (float, [0, 1])

####################
## Factor scores ##
####################

# Select metrics
CLCV = dplyr::select(results_main,m1_0, m2_0, m3_0, m4_0, m7_0, m8_0, m11_0, m12_0, m19_0) # Color variability, clutter
G = dplyr::select(results_main,m21_0, m21_2, m21_4, m21_6, m21_8) # Grid quality

# FACTOR ANALYSIS --> number of appropriate factors assessed through n_factors first
#For instance, n_factors(G)
CLCV_fa <- factanal(CLCV, factors =3 , rotation = "varimax", scores="regression", )
G_fa <- factanal(G, factors =1 , rotation = "varimax", scores="regression", )

# This just combines the regression scores to the main dataframe
rownames(results_main) <- 1:nrow(results_main)
CLCV_fs <- data.frame(CLCV_fa$scores)
G_fs <- data.frame(G_fa$scores)

results_main$rowname <- rownames(results_main)
CLCV_fs$rowname <- rownames(CLCV_fs)
colnames(G_fs) <- c('Factor4')
G_fs$rowname <- rownames(G_fs)

results_main = left_join(results_main, CLCV_fs, by = "rowname")
results_main = left_join(results_main, G_fs, by = "rowname")

factors = dplyr::select(results_main, m6_0, m5_0, Factor1, Factor2, Factor3, Factor4, img_name)

# Save results with search times
df = left_join(df, factors, by = "img_name")
write.csv(df, "output/data/vsgui10k_search_times_with_factors.csv")
write.csv(results_main, "output/data/vsgui10k_aim_results_with_factors.csv")