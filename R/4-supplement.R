##
library(tidyverse)
library(lme4)
library(emmeans)
library(lmerTest)
library(gridExtra)
library(MASS)
library(ggplot2)
library(performance)
library(texreg) ## for latex-table, otherwise not needed
library(dplyr)
library(sjPlot)
library(cowplot)
library(xtable)
library(ggpubr)
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

# This is search time data with factors
df_factors <- read.csv("output/data/vsgui10k_search_times_with_factors.csv", header = T) 

# Tweak column names for clarity
colnames(df_factors)[which(names(df_factors) == "m6_0")] <- "ContourCongestion"
colnames(df_factors)[which(names(df_factors) == "m5_0")] <- "FigureGroundContrast"

df$cue <- relevel(df$cue, ref = "Image")   # Change baselines here to check for alternatives
df$category <- relevel(df$category, ref = "Web")

####################
## Model fitting ##
####################

## Interaction model with set size (Fig. 17 in supplement)
fit_interaction_set <- lmer(search_time ~
                          set_size * absent +
                          category* absent +
                          cue * absent +
                          (1 | pid) +
                          (1 | img_name),
                        data = df)

## Visual complexity model with set size (Fig. 18 in supplement)
fit_set_aesthetics <- lmer(search_time ~
                             Factor1 +  
                             Factor2 +
                             Factor3 +
                             Factor4  +
                             FigureGroundContrast +
                             ContourCongestion +
                             set_size +
                             (1 | pid)+
                             (1 | img_name),
                           data = df_factors)

## Controlled variable model with GUI size (Fig. 19 in supplement)
fit_controlled_image <- lmer(search_time ~
                               scale_x +
                               scale_y +
                               cue +
                               absent +
                               (1 | pid) +
                               (1 | img_name),
                             data = df)

## Target model (Fig. 20 in supplement)
fit_target <- lmer(search_time ~
                     category +
                     cue +
                     tgt_y_center_from_fixation_cross +
                     tgt_x_center_from_fixation_cross+
                     (1 | pid) +
                     (1 | img_name),
                   data = df %>% filter(absent == 'False'))

##################
## Plot Fig. 17 ##
##################

set_order = c(2, 10, 6, 5, 7, 11, 1, 3, 8, 4, 9)
set_labels = c("Target absent × Category=Mobile", 
               "Category=Mobile", 
               "Target absent × Category=Desktop", 
               "Category=Desktop",
               "Set size",
               "Target absent x Cue=Text+color",
               "Target absent × Set size",
               "Cue=Text",
               "Cue=Text+color",
               "Target absent × Cue=Text",
               "Target absent")
set_groups = c("A", "A", "B", "B", "A", "A", "A", "B","B", "A","C")

pforest <- plot_model(fit_interaction_set, type="est", title="", axis.title="Estimates (s)", show.values=TRUE, show.p=TRUE, show.ci=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-10,25), order.terms = set_order, axis.labels=set_labels, group.terms=set_groups, colors = "black",  vline.color = "#696969")

pforest_std <- plot_model(fit_interaction_set, type="std", title="", axis.title="Estimates (standardized)", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-3,2), order.terms = set_order, axis.labels = set_labels, group.terms=set_groups, colors = "black",  vline.color = "#696969") 

# Save
path = "output/figs/supplement/supp_fig17.pdf"
pdf(path, width=10, height=9)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

##################
## Plot Fig. 18 ##
##################

aesthetics_labels = c("Set size", "Factor 2 (Color variability 2,5)", "Figure-ground contrast", "Factor 3 (Color variability 1,3)", "Factor 4 (Grid quality)",  "Factor 1 (Visual clutter)", "Contour congestion")
aesthetics_order = c(6, 1, 4, 3, 5, 2,7)
aesthetics_group = c("A","C","A","C","C", "A", "A")
aesthetics_group_fixations = c("A","C","A","A","C", "A", "A")

pforest <- plot_model(fit_set_aesthetics, type="est", title="", axis.title="Estimates (s)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-1,20), sort.est=TRUE, order.terms = aesthetics_order, group.terms = aesthetics_group,  axis.labels = aesthetics_labels,  vline.color = "#696969")

pforest_std <- plot_model(fit_set_aesthetics, type="std", title="", axis.title="Estimates (standardized)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-1,4), sort.est=TRUE, order.terms = aesthetics_order, group.terms = aesthetics_group, axis.labels = aesthetics_labels,  vline.color = "#696969")

path = "output/figs/supplement/supp_fig18.pdf"
pdf(path, width=10, height=3)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()


##################
## Plot Fig. 19 ##
##################


# Plot for search times with image
pforest <- plot_model(fit_controlled_image, type="est", title="", axis.title="Estimates (s)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-3,8), sort.est=TRUE, order.terms=c(5,3,4,2,1), axis.labels = c("GUI height", "GUI width", "Cue=Text+color", "Cue=Text", "Target absent"),  vline.color = "#696969")
pforest_std <- plot_model(fit_controlled_image, type="std", title="", axis.title="Estimates (standardized)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-2,2), order.terms=c(5,3,4,2,1), axis.labels = c("GUI height", "GUI width", "Cue=Text+color", "Cue=Text", "Target absent"),  vline.color = "#696969")

pdf("output/figs/supplement/supp_fig19.pdf", width=10, height=3)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

##################
## Plot Fig. 20 ##
##################


set_theme(
  base = theme_bw(),
  theme.font = "sans",
  axis.title.color = "black",
  geom.label.color = "black",
  axis.textcolor.x = "black",
  axis.textcolor.y = "black",
  axis.textsize.x = 1.2,
  axis.textsize.y = 1.2,
  axis.title.size = 1.3,
)

labels = c( "Category=Mobile", "Category=Desktop", "Distance from fixation cross (Y)", "Distance from fixation cross (X)",  "Cue=Text+color","Cue=Text")
order = c(4,3,6,5,1,2)
group = c("B","B","A","A","C", "A")

# Plot for search times
pforest <- plot_model(fit_target, type="est", title="", axis.title="Estimates (s)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-3,8), axis.labels = labels, order.terms = order, group.terms=group,  vline.color = "#696969")
pforest_std <- plot_model(fit_target, type="std", title="", axis.title="Estimates (standardized)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-3,8), order.terms = order, axis.labels=labels, group.terms=group,  vline.color = "#696969")

pdf("output/figs/supplement/supp_fig20.pdf", width=10, height=3)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()
