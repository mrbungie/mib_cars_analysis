library(arrow)
library(dplyr)
library(forcats)
library(ggplot2)
library(patchwork)
library(scales)

root_dir <- normalizePath(".", mustWork = TRUE)
data_path <- file.path(root_dir, "data", "intermediate", "cleaned_data.parquet")
asset_path <- file.path(root_dir, "slidedeck", "assets", "eda_distribution_by_type_subtype_region.png")

short_label <- function(x) {
  dplyr::recode(
    x,
    "Batteries & Accessories" = "Batteries & Acc.",
    "Exterior Accessories" = "Exterior Acc.",
    "Garage & Car Care" = "Garage & Care",
    "Interior Accessories" = "Interior Acc.",
    "Performance Parts" = "Perf. Parts",
    "Replacement Parts" = "Replacement",
    "Towing & Hitches" = "Towing & Hitch.",
    .default = x
  )
}

build_summary <- function(df, col_name, panel_name) {
  df %>%
    group_by(category = .data[[col_name]]) %>%
    summarise(
      opportunities = n(),
      win_rate = mean(won_flag),
      p25 = quantile(`Opportunity Amount USD`, 0.25, names = FALSE),
      median_amount = median(`Opportunity Amount USD`),
      p75 = quantile(`Opportunity Amount USD`, 0.75, names = FALSE),
      .groups = "drop"
    ) %>%
    arrange(desc(win_rate), desc(median_amount)) %>%
    mutate(
      panel = panel_name,
      label = short_label(category),
      panel_label = factor(paste(panel, label, sep = "___"), levels = rev(paste(panel, label, sep = "___")))
    )
}

base_theme <- theme_minimal(base_family = "Helvetica") +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "#E2E8F0", linewidth = 0.4),
    axis.title = element_text(size = 11.2, color = "#334155"),
    axis.text = element_text(size = 10, color = "#0F172A"),
    strip.text.y.left = element_text(size = 12.2, face = "bold", color = "#0F172A", angle = 0),
    strip.background = element_blank(),
    plot.title = element_text(size = 14.5, face = "bold", color = "#0F172A", hjust = 0),
    plot.subtitle = element_text(size = 10.5, color = "#475569", hjust = 0),
    plot.margin = margin(8, 8, 8, 8)
  )

df <- read_parquet(data_path) %>%
  select(
    `Supplies Group`,
    `Supplies Subgroup`,
    Region,
    `Opportunity Result`,
    `Opportunity Amount USD`
  ) %>%
  filter(!if_any(everything(), is.na)) %>%
  filter(`Opportunity Result` %in% c("Won", "Loss")) %>%
  mutate(won_flag = `Opportunity Result` == "Won")

summary_df <- bind_rows(
  build_summary(df, "Supplies Group", "Type"),
  build_summary(df, "Supplies Subgroup", "Subtype"),
  build_summary(df, "Region", "Region")
) %>%
  mutate(panel = factor(panel, levels = c("Region", "Type", "Subtype")))

mix_plot <- ggplot(summary_df, aes(y = panel_label)) +
  geom_col(aes(x = 1), width = 0.66, fill = "#E2E8F0", color = NA) +
  geom_col(aes(x = win_rate), width = 0.66, fill = "#2563EB", color = NA) +
  geom_text(
    data = summary_df,
    aes(x = pmax(win_rate - 0.03, 0.03), y = panel_label, label = percent(win_rate, accuracy = 1)),
    inherit.aes = FALSE,
    color = "white",
    fontface = "bold",
    size = 3.2
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    labels = percent_format(accuracy = 1),
    expand = c(0, 0)
  ) +
  scale_y_discrete(labels = function(x) sub(".*___", "", x)) +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(
    title = "Conversion mix",
    x = "Opportunity share",
    y = NULL
  ) +
  base_theme +
  theme(
    strip.placement = "outside",
    axis.text.y = element_text(size = 9.3),
    axis.line.x = element_line(color = "#CBD5E1"),
    axis.ticks.x = element_blank(),
    plot.margin = margin(8, 18, 8, 8)
  )

amount_plot <- ggplot(summary_df, aes(y = panel_label)) +
  geom_segment(
    aes(x = p25, xend = p75, yend = panel_label),
    linewidth = 1.6,
    color = "#94A3B8",
    lineend = "round"
  ) +
  geom_point(aes(x = median_amount), size = 2.8, color = "#0F766E") +
  scale_x_continuous(
    limits = c(0, 200000),
    labels = label_dollar(scale = 1 / 1000, suffix = "K", accuracy = 1),
    expand = c(0, 0)
  ) +
  scale_y_discrete(labels = function(x) sub(".*___", "", x)) +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(
    title = "Typical deal size",
    x = "Deal amount in USD",
    y = NULL
  ) +
  base_theme +
  theme(
    strip.text.y.left = element_blank(),
    strip.placement = "outside",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.x = element_line(color = "#CBD5E1"),
    axis.ticks.x = element_blank(),
    plot.margin = margin(8, 8, 8, 18)
  )

combined <- mix_plot + amount_plot +
  plot_layout(widths = c(1.05, 0.95))

ggsave(asset_path, combined, width = 13.6, height = 9.3, dpi = 220, bg = "white")
cat(sprintf("saved: %s\n", asset_path))
