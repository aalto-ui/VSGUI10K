## Controlled-variable model

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
    # There seems to be a bad entry; remove it
    filter(search_time > 0) 

df$cue <- relevel(df$cue, ref = "Image")   # Change baselines here to check for alternatives
df$category <- relevel(df$category, ref = "Web")

path <- "output/figs/1-controlled" # Save figs here
dir.create(path)

#####################
## Model fitting ##
#####################

# Controlled variable model (in main paper)
fit_controlled <- lmer(search_time~
                         category +
                         cue +
                         absent +
                         (1 | pid) +
                         (1 | img_name),
                        data = df)

# Controlled variable model with fixations as DV (in supplement)
fit_controlled_fixations <- lmer(n_fixations ~
                         category +
                         cue +
                         absent +
                         (1 | pid) +
                         (1 | img_name),
                       data = df)

##########
## Plot ##
##########

set_theme(
  base = theme_light(),
  theme.font = "sans",
  axis.title.color = "black",
  geom.label.color = "black",
  axis.textcolor.x = "black",
  axis.textcolor.y = "black",
  axis.textsize.x = 1.2,
  axis.textsize.y = 1.2,
  axis.title.size = 1.3,
)

colors <- c("#000080", "#a40000") 

# Plot for search times
pforest <- plot_model(fit_controlled, type = "est", title = "", axis.title = "Estimates (s)", show.values = TRUE, show.p = TRUE, value.offset = c(0.35, 0.35), axis.lim = c(-3, 8), sort.est = TRUE, axis.labels = c("Category=Mobile", "Category=Desktop", "Cue=Text+color", "Cue=Text", "Target absent"), colors = colors, vline.color = "#696969") 
           
pforest_std <- plot_model(fit_controlled, type = "std", title = "", axis.title = "Estimates (standardized)", show.values = TRUE, show.p = TRUE, value.offset = c(0.35, 0.35), axis.lim = c(-2, 2), sort.est = TRUE, axis.labels = c("Category=Mobile", "Category=Desktop", "Cue=Text+color", "Cue=Text", "Target absent"), colors = colors, vline.color = "#696969")

# Add a custom legend
pforest <- pforest + 
  scale_color_manual(name = "", 
                     values = colors, 
                     labels = c("Decrease", "Increase")) +
  guides(color = guide_legend()) +
  theme(legend.position = "bottom")

# Add a custom legend
pforest_std <- pforest_std + 
  scale_color_manual(name = "", 
                     values = colors, 
                     labels = c("Decrease", "Increase")) +
  guides(color = guide_legend()) +
  theme(legend.position = "bottom")

# Save
pdf("output/figs/1-controlled/fig3.pdf", width = 10, height = 3)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol = 2, labels = c("A", "B"))
dev.off()

# Plot for fixations
pforest <- plot_model(fit_controlled_fixations, type = "est", title = "", axis.title = "Estimates (fixations)", colors = "black", show.values = TRUE, show.p = TRUE, value.offset = c(0.35, 0.35), axis.lim = c(-10, 22), sort.est = TRUE, axis.labels = c("Category=Mobile", "Category=Desktop", "Cue=Text+color", "Cue=Text", "Target absent"), vline.color = "#696969")
pforest_std <- plot_model(fit_controlled_fixations, type = "std", title = "", axis.title = "Estimates (standardized)", colors = "black", show.values = TRUE, show.p = TRUE, value.offset = c(0.35, 0.35), axis.lim = c(-2, 2), sort.est = TRUE, axis.labels = c("Category=Mobile", "Category=Desktop", "Cue=Text+color", "Cue=Text", "Target absent"), vline.color = "#696969")

# Save
pdf("output/figs/1-controlled/supp_fig13.pdf", width = 10, height = 3)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol = 2, labels = c("A", "B"))
dev.off()

###################
## Print tabular ##
###################

# Use this to check SDs, etc.

tab_model(fit_controlled, show.reflvl = TRUE, show.se = TRUE, show.std = TRUE, show.icc = TRUE, dv.labels = c("Controlled Model (linear)"))
