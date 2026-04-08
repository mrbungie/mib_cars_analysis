library(arrow)
library(ggplot2)
library(dplyr)

root_dir <- normalizePath('.', mustWork = TRUE)
data_path <- file.path(root_dir, 'data', 'intermediate', 'cleaned_data.parquet')
asset_path <- file.path(root_dir, 'slidedeck', 'assets', 'annex_numeric_correlation_heatmap.png')

df <- read_parquet(data_path)

selected <- df %>%
  select(
    `Elapsed Days In Sales Stage`,
    `Sales Stage Change Count`,
    `Total Days Identified Through Closing`,
    `Total Days Identified Through Qualified`,
    `Opportunity Amount USD`,
    `Ratio Days Identified To Total Days`,
    `Ratio Days Validated To Total Days`,
    `Ratio Days Qualified To Total Days`,
    opportunity_amount_weirdness,
    problem_count
  )

label_map <- c(
  `Elapsed Days In Sales Stage` = 'Elapsed days in stage',
  `Sales Stage Change Count` = 'Stage changes',
  `Total Days Identified Through Closing` = 'Days: identified → closing',
  `Total Days Identified Through Qualified` = 'Days: identified → qualified',
  `Opportunity Amount USD` = 'Opportunity amount',
  `Ratio Days Identified To Total Days` = 'Ratio: identified / total',
  `Ratio Days Validated To Total Days` = 'Ratio: validated / total',
  `Ratio Days Qualified To Total Days` = 'Ratio: qualified / total',
  opportunity_amount_weirdness = 'Amount weirdness',
  problem_count = 'Problem count'
)

cor_mat <- cor(selected, use = 'pairwise.complete.obs')
plot_df <- as.data.frame(as.table(cor_mat)) %>%
  rename(var_x = Var1, var_y = Var2, correlation = Freq) %>%
  mutate(
    var_x = factor(label_map[as.character(var_x)], levels = label_map[colnames(selected)]),
    var_y = factor(label_map[as.character(var_y)], levels = rev(label_map[colnames(selected)]))
  )

p <- ggplot(plot_df, aes(var_x, var_y, fill = correlation)) +
  geom_tile(color = 'white', linewidth = 0.8) +
  geom_text(aes(label = sprintf('%.2f', correlation)), size = 3.1, color = '#0F172A') +
  scale_fill_gradient2(
    low = '#B91C1C',
    mid = '#F8FAFC',
    high = '#1D4ED8',
    midpoint = 0,
    limits = c(-1, 1),
    name = 'Correlation'
  ) +
  labs(
    title = 'Correlation heatmap of numeric variables',
    subtitle = 'Pearson correlation on the cleaned modelling dataset',
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_family = 'Helvetica') +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 35, hjust = 1, size = 9, color = '#0F172A'),
    axis.text.y = element_text(size = 9, color = '#0F172A'),
    plot.title = element_text(size = 14, face = 'bold', color = '#0F172A'),
    plot.subtitle = element_text(size = 10, color = '#475569'),
    legend.position = 'right'
  )

ggsave(asset_path, p, width = 11, height = 8.2, dpi = 220, bg = 'white')
cat(sprintf('saved: %s\n', asset_path))
