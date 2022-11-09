library(ggplot2)
library(stringr)


# # Root scores
data <- read.csv("../results/table_roots.csv")

custom_order = c("max_width", "node_count", "not_solved_count")

data$X3 <- round(data$X3, digits=2)

data$X1 <- factor(data$X1, levels=rev(ordered(custom_order)))

ggplot(data, aes(x=X0, y=X1, fill=X2)) +
geom_tile(color="black") +
geom_text(aes(label = ifelse(X3 < 0.01, "< 0.01", X3)),
          color = "white", size = 4) +
theme_bw() +
  scale_fill_gradient(low="#3B9AB2", high="#9dccd8") +
# scale_fill_gradient(low="#2068ab", high="#56b1f7") +
scale_x_discrete(labels = c("ra" = "RAscore", "sa" =  "SAscore", "sc" = "SCScore", "syba" = "SYBA") ) +
scale_y_discrete(labels = str_wrap(c("not_solved_count" = "Number of not solved leaves", "node_count" =  "Number of nodes", "max_width" = "Maximum width"),
                                   width=10)) +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
labs(fill = str_wrap("Spearman correlation", width=10)) 

ggsave("heatmap_root_scores.pdf",
       width = 5, height = 2.5)

# # Siblings scores
library(colorspace)

custom_order = c("sa min",
                 "sa max",
                 "sa avg",
                 "sc min",
                 "sc max",
                 "sc avg",
                 "ra max",
                 "ra min",
                 "ra avg",
                 "syba max",
                 "syba min",
                 "syba avg")

data <- read.csv("../results/table_siglings.csv")

# data$X2 <- log10(data$X2)
data$X2 <- round(data$X2, digits=3)

# data$X1 <- factor(data$X1)
data$X1 <- factor(data$X1, levels=rev(ordered(custom_order)))

# custom_colours = c("#8b9b7c", "#edda78", "#de8a5a")
# custom_colours = c("#0073C2", "#868686", "#EFC000") # JCO
# custom_colours = c("#D2AF81", "#8A9197", "#709AE1") # Simpsons

# library(wesanderson)
my_wes_palette <- function(pal, n, type = c("discrete", "continuous")) {
  type <- match.arg(type)
  
  # pal <- wes_palettes[[name]]
  if (is.null(pal))
    stop("Palette not found.")
  
  if (missing(n)) {
    n <- length(pal)
  }
  
  if (type == "discrete" && n > length(pal)) {
    stop("Number of requested colors greater than what palette can offer")
  }
  
  out <- switch(type,
                continuous = grDevices::colorRampPalette(pal)(n),
                discrete = pal[1:n]
  )
  structure(out, class = "palette")
}

#' #' @export
#' #' @importFrom graphics rect par image text
#' #' @importFrom grDevices rgb
#' print.palette <- function(x, ...) {
#'   n <- length(x)
#'   old <- par(mar = c(0.5, 0.5, 0.5, 0.5))
#'   on.exit(par(old))
#'   
#'   image(1:n, 1, as.matrix(1:n), col = x,
#'         ylab = "", xaxt = "n", yaxt = "n", bty = "n")
#'   
#'   rect(0, 0.9, n + 1, 1.1, col = rgb(1, 1, 1, 0.8), border = NA)
#'   text((n + 1) / 2, 1, labels = attr(x, "name"), cex = 1, family = "serif")
#' }

# custom_colours <- wes_palette("Rushmore1", 100, type = "continuous")

# Zissou1 with removed red
custom_colours <- my_wes_palette(c("#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00"),
                              100, type = "continuous")

a = ggplot(data, aes(x=X0, y=X1, fill=X2)) +
  geom_tile(color="black") +
  geom_text(aes(label = ifelse(X2 < 0.001, "< 0.001", X2)),
           color = "white", size = 4) +
  theme_bw() +
  scale_fill_gradientn(
    colours = custom_colours,
    # values = c(0, 0.01, 1),
    breaks = c(0, 0.01, 1),  trans= scales::pseudo_log_trans(sigma = 0.001)
    #guide="none"
  ) +
  scale_x_discrete(breaks = c("Internal - Not Solved0", "Solved - Not Solved1",
                        "Internal - Not Solved2"),
                   labels = c("Internal - Not Solved0" = "Internal - Not Solved\nall molecules",
                                       "Solved - Not Solved1" =  "Solved - Not Solved\nall molecules",
                                       "Internal - Not Solved2" = "Internal - Not Solved\nexpandable molecules")
                                     ) +
  scale_y_discrete(breaks = custom_order,
                   labels = str_wrap(c("sa min" = "SAscore maximum",
                                       "sa max" = "SAscore minimum",
                                       "sa avg" = "SAscore mean",
                                       "sc min" = "SCScore maximum",
                                       "sc max" = "SCScore minimum",
                                       "sc avg" = "SCScore mean",
                                       "ra max" = "RAscore maximum",
                                       "ra min" = "RAscore minimum",
                                       "ra avg" = "RAscore mean",
                                       "syba max" = "SYBA maximum",
                                       "syba min" = "SYBA minimum", # Warning, min max swapped
                                       "syba avg" = "SYBA mean"
                   ),
                   width=10)
  ) +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(fill = str_wrap("p-value", width=10)) +
  guides(fill="none")

a

# # Parent-child scores
data <- read.csv("../results/table_parental.csv")

# data$X2 <- log10(data$X2)
data$X2 <- round(data$X2, digits=3)


data$X1 <- factor(data$X1, levels=rev(ordered(custom_order)))


b = ggplot(data, aes(x=X0, y=X1, fill=X2)) +
  geom_tile(color="black") +
  geom_text(aes(label = ifelse(X2 < 0.001, "< 0.001", X2)),
            color = "white", size = 4) +
  theme_bw() +
  scale_fill_gradientn(
    colours = custom_colours,
    # values = c(0, 0.01, 1),
    breaks = c(0.01, 1), trans= scales::pseudo_log_trans(sigma = 0.001)
    #guide="none"
  ) +
  scale_x_discrete(breaks = c("Internal - Not Solved0", "Internal - Not Solved1"),
                   labels = c("Internal - Not Solved0" = "Internal - Not Solved\nall molecules",
                              "Internal - Not Solved1" = "Internal - Not Solved\nexpandable molecules")
  ) +
  scale_y_discrete(breaks = custom_order,
                   labels = str_wrap(c("sa min" = "SAscore maximum",
                                       "sa max" = "SAscore minimum",
                                       "sa avg" = "SAscore mean",
                                       "sc min" = "SCScore maximum",
                                       "sc max" = "SCScore minimum",
                                       "sc avg" = "SCScore mean",
                                       "ra max" = "RAscore maximum",
                                       "ra min" = "RAscore minimum",
                                       "ra avg" = "RAscore mean",
                                       "syba max" = "SYBA maximum",
                                       "syba min" = "SYBA minimum", # Warning, min max swapped
                                       "syba avg" = "SYBA mean"
                                       ),
                                     width=10)
  ) +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(fill = str_wrap("p-value", width=10)) 

b

# # Merge up a and b
library(cowplot)

plot_grid(a, b, labels=c("A", "B"), rel_widths = c(7,6))

ggsave("heatmaps_siblings_parental.pdf",
       width = 10, height = 4)


# # Heatmap of AUC
library(dplyr)
data.in <- read.csv("../results/table_aucs.csv")

# data$X2 <- log10(data$X2)
data.in$X2 <- round(data.in$X2, digits=2)

custom_order = rev(ordered(c("min", "max", "avg")))

data.in$X1 <- factor(data.in$X1, levels=custom_order)


# Just for easier checking
data = data.in

data.in %>%
ggplot(aes(x=X0, y=X1, fill=X2, group=X3)) +
  geom_tile(color="black", position = position_dodge(), width=0.9, height=0.9) +
  # geom_text(aes(label = ifelse(X2 < 0.001, "< 0.001", X2)),
  #           color = "white", size = 4, position = position_dodge(0.9)) +
  theme_bw() +
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  labs(fill = str_wrap("AUC", width=10)) +
  # scale_fill_distiller(palette = "Blues", direction = 1)
  scale_fill_gradient(low="#b0d6e0", high="#3B9AB2") +
  scale_x_discrete(labels = c("ra" = "RAscore", "sa" =  "SAscore", "sc" = "SCScore", "syba" = "SYBA") ) +
  scale_y_discrete(breaks = custom_order,
                   labels = str_wrap(rev(ordered(c("min" = "Minimum", "max" =  "Maximum", "avg" = "Mean"))),
                                     width=10))
  # geom_text(data = data.in %>% filter(X0 == "syba", X1 == "min"),
  #           aes(label=X3))


ggsave("heatmap_aucs.pdf",
       width = 6, height = 3)
