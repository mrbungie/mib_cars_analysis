library(arrow)
library(dplyr)
library(ggplot2)
library(readxl)
library(scales)

root_dir <- normalizePath('.', mustWork = TRUE)
regression_path <- file.path(root_dir, 'slidedeck', 'data', 'regression_model_report.xlsx')
tagged_path <- file.path(root_dir, 'data', 'intermediate', 'df_tagged_full.parquet')
forecast_asset <- file.path(root_dir, 'slidedeck', 'assets', 'annex_forecast_bias.png')
issues_asset <- file.path(root_dir, 'slidedeck', 'assets', 'annex_data_quality_issue_types.png')

theme_slide <- theme_minimal(base_family = 'Helvetica') +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = '#E2E8F0', linewidth = 0.4),
    axis.title = element_text(size = 10.5, color = '#334155'),
    axis.text = element_text(size = 9.5, color = '#0F172A'),
    plot.title = element_text(size = 13, face = 'bold', color = '#0F172A'),
    plot.subtitle = element_text(size = 9.5, color = '#475569'),
    plot.margin = margin(8, 12, 8, 8)
  )

forecast_summary <- read_excel(regression_path, sheet = 'forecast_summary')
actual_total <- forecast_summary$value[forecast_summary$metric == 'actual_total_amount']
pred_total <- forecast_summary$value[forecast_summary$metric == 'predicted_total_amount']
mae_value <- forecast_summary$value[forecast_summary$metric == 'mean_absolute_error']
mdape_value <- forecast_summary$value[forecast_summary$metric == 'median_absolute_percentage_error']
bias_pct <- (pred_total / actual_total) - 1
gap_value <- actual_total - pred_total

forecast_df <- tibble::tibble(
  series = factor(c('Actual total', 'Predicted total'), levels = c('Actual total', 'Predicted total')),
  amount = c(actual_total, pred_total)
)

forecast_plot <- ggplot(forecast_df, aes(x = series, y = amount, fill = series)) +
  geom_col(width = 0.58, show.legend = FALSE) +
  geom_text(aes(label = label_dollar(scale = 1e-9, suffix = 'B', accuracy = 0.01)(amount)), vjust = -0.45, size = 3.6, fontface = 'bold', color = '#0F172A') +
  annotate('segment', x = 1, xend = 2, y = actual_total * 0.93, yend = actual_total * 0.93, color = '#DC2626', linewidth = 1.1) +
  annotate('text', x = 1.5, y = actual_total * 0.965, label = paste0('Gap: ', label_dollar(scale = 1e-6, suffix = 'M', accuracy = 1)(gap_value), ' (', percent(abs(bias_pct), accuracy = 1), ' under)'), color = '#DC2626', size = 3.5, fontface = 'bold') +
  scale_fill_manual(values = c('Actual total' = '#93C5FD', 'Predicted total' = '#2563EB')) +
  scale_y_continuous(labels = label_dollar(scale = 1e-9, suffix = 'B', accuracy = 0.1), expand = expansion(mult = c(0, 0.12))) +
  labs(title = NULL, subtitle = NULL, x = NULL, y = 'Held-out portfolio total') +
  theme_slide +
  theme(
    axis.line.y = element_line(color = '#CBD5E1'),
    axis.line.x = element_line(color = '#CBD5E1')
  )

ggsave(forecast_asset, forecast_plot, width = 7.4, height = 4.5, dpi = 220, bg = 'white')

df <- read_parquet(tagged_path)
issue_counts <- tibble::tibble(
  issue = c(
    'Stage-ratio inconsistency',
    'Amount weirdness > 75th pct',
    'Only identified stage',
    'Zero total days',
    'Zero amount opportunity',
    'Amount outlier',
    'Total-days outlier',
    'Partially repeated row',
    'Fully repeated row'
  ),
  count = c(
    sum(df$flag_ratio_problem, na.rm = TRUE),
    sum(df$flag_weirdness_over_75th_pct, na.rm = TRUE),
    sum(df$flag_only_identified, na.rm = TRUE),
    sum(df$flag_0_days, na.rm = TRUE),
    sum(df$flag_zero_opportunity_amount, na.rm = TRUE),
    sum(df$flag_outlier_opportunity_amount, na.rm = TRUE),
    sum(df$flag_outlier_total_days, na.rm = TRUE),
    sum(df$flag_partially_repeated_row, na.rm = TRUE),
    sum(df$flag_totally_repeated_row, na.rm = TRUE)
  )
) %>%
  arrange(desc(count)) %>%
  mutate(issue = factor(issue, levels = rev(issue)))

issues_plot <- ggplot(issue_counts, aes(x = count, y = issue)) +
  geom_col(fill = '#94A3B8', width = 0.64) +
  geom_text(aes(label = comma(count)), hjust = -0.08, size = 3.5, color = '#0F172A', fontface = 'bold') +
  scale_x_continuous(labels = comma, expand = expansion(mult = c(0, 0.12))) +
  labs(title = NULL, subtitle = NULL, x = 'Rows flagged', y = NULL) +
  theme_slide +
  theme(
    axis.line.x = element_line(color = '#CBD5E1'),
    axis.text.y = element_text(size = 9.8)
  )

ggsave(issues_asset, issues_plot, width = 8.2, height = 4.7, dpi = 220, bg = 'white')

cat(sprintf('saved: %s\n', forecast_asset))
cat(sprintf('saved: %s\n', issues_asset))
cat(sprintf('forecast gap: %s | mae: %s | mdape: %s\n',
            label_dollar(scale = 1e-6, suffix = 'M', accuracy = 1)(gap_value),
            label_dollar(scale = 1/1000, suffix = 'K', accuracy = 0.1)(mae_value),
            percent(mdape_value, accuracy = 1)))
