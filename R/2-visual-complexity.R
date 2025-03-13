# Visual-complexity model
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

# This is search time data with factors
df <- read.csv("output/data/vsgui10k_search_times_with_factors.csv", header = T) 

# Tweak column names for clarity
colnames(df)[which(names(df) == "m6_0")] <- "ContourCongestion"
colnames(df)[which(names(df) == "m5_0")] <- "FigureGroundContrast"

path = "output/figs/2-complexity" # Save here
dir.create(path)

####################
## Model fitting ##
####################

## Visual complexity model (in main paper)
fit_complexity <- lmer(search_time ~
                          Factor1 +
                          Factor2 +
                          Factor3 +
                          Factor4  +
                          FigureGroundContrast +
                          ContourCongestion +
                          (1 | pid)+
                          (1 | img_name),
                          data = df)

## Visual complexity model with fixations as DV
fit_complexity_fixations <- lmer(n_fixations ~
                         Factor1 +
                         Factor2 +
                         Factor3 +
                         Factor4  +
                         FigureGroundContrast +
                         ContourCongestion +
                         (1 | pid)+
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

## Order of terms
# Factor1 +
# Factor2 +
# Factor3 +
# Factor4  +
# m5_0 +
# m6_0 +

aesthetics_labels = c("Factor 2 (Color variability 2,5)", "Figure-ground contrast", "Factor 3 (Color variability 1,3)", "Factor 4 (Grid quality)",  "Factor 1 (Visual clutter)", "Contour congestion")
aesthetics_order = c(6, 1, 4, 3, 5, 2)
aesthetics_group = c("A","C","A","A","C", "A")


group.colors <- c(A = "#000080", B = "#a40000", C ="azure4")

pforest <- plot_model(fit_complexity, type="est", title="", axis.title="Estimates (s)", colors=group.colors, show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-1,6), sort.est=TRUE, order.terms = aesthetics_order, axis.labels = aesthetics_labels, group.terms = aesthetics_group, vline.color = "#696969")

pforest_std <- plot_model(fit_complexity, type="std", title="", axis.title="Estimates (standardized)", colors=group.colors, show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-1,1), sort.est=TRUE, order.terms = aesthetics_order, axis.labels = aesthetics_labels, group.terms = aesthetics_group, vline.color = "#696969")

# Add a custom legend
pforest <- pforest + 
  scale_color_manual(name = "", 
                     values =c("#a40000", "azure4"), 
                     labels = c("Increase", "Not significant")) +
  guides(color = guide_legend()) +
  theme(legend.position = "bottom")

# Add a custom legend
pforest_std <- pforest_std + 
  scale_color_manual(name = "", 
                     values =c("#a40000", "azure4"), 
                     labels = c("Increase", "Not significant")) +
  guides(color = guide_legend()) +
  theme(legend.position = "bottom")

# Save
path = "output/figs/2-complexity/fig7.pdf"
pdf(path, width=10, height=4)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

# Plot for fixations
pforest <- plot_model(fit_complexity_fixations, type="est", title="", axis.title="Estimates (fixations)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-10,22), sort.est=TRUE, order.terms = aesthetics_order, axis.labels = aesthetics_labels, group.terms = aesthetics_group, vline.color = "#696969")
pforest_std <- plot_model(fit_complexity_fixations, type="std", title="", axis.title="Estimates (standardized)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-2,2), sort.est=TRUE, order.terms = aesthetics_order, axis.labels = aesthetics_labels, group.terms = aesthetics_group, vline.color = "#696969")

# Save
path = "output/figs/2-complexity/supp_fig14.pdf"
pdf(path, width=10, height=4)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

###################
## Print tabular ##
###################

# Use this to check SDs, etc.

tab_model(fit_complexity, show.reflvl = TRUE, show.se = TRUE, show.std=TRUE,show.icc=TRUE, dv.labels = c("Complexity Model (linear)"))
