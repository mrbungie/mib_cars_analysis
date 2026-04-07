library(arrow)
library(dplyr)
library(ggplot2)
library(scales)

root_dir <- normalizePath(".", mustWork = TRUE)
data_path <- file.path(root_dir, "data", "intermediate", "cleaned_data.parquet")
client_asset <- file.path(root_dir, "slidedeck", "assets", "eda_client_segmentation.png")
funnel_asset <- file.path(root_dir, "slidedeck", "assets", "eda_two_phase_funnel.png")

theme_slide <- theme_minimal(base_family = "Helvetica") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E2E8F0", linewidth = 0.4),
    axis.title = element_text(size = 10.5, color = "#334155"),
    axis.text = element_text(size = 9.5, color = "#0F172A"),
    plot.title = element_text(size = 13, face = "bold", color = "#0F172A"),
    plot.subtitle = element_text(size = 9.5, color = "#475569"),
    plot.margin = margin(8, 12, 8, 8)
  )

df <- read_parquet(data_path) %>%
  filter(`Opportunity Result` %in% c("Won", "Loss")) %>%
  mutate(won_flag = `Opportunity Result` == "Won")

client_df <- df %>%
  group_by(segment = `Client Size By Revenue (USD)`) %>%
  summarise(
    opportunities = n(),
    win_rate = mean(won_flag),
    .groups = "drop"
  ) %>%
  mutate(
    segment = factor(
      segment,
      levels = c("100K or less", "100K to 250K", "250K to 500K", "500K to 1M", "More than 1M")
    )
  ) %>%
  arrange(segment)

max_client_count <- max(client_df$opportunities)
client_scale <- max_client_count / 0.40

client_plot <- ggplot(client_df, aes(x = segment)) +
  geom_col(aes(y = opportunities), fill = "#BFDBFE", width = 0.68) +
  geom_text(
    aes(y = opportunities, label = comma(opportunities)),
    vjust = -0.45,
    size = 3.2,
    color = "#1E3A8A",
    fontface = "bold"
  ) +
  geom_line(
    aes(y = win_rate * client_scale, group = 1),
    color = "#0F172A",
    linewidth = 1.2
  ) +
  geom_point(aes(y = win_rate * client_scale), color = "#0F172A", size = 2.4) +
  geom_text(
    aes(y = win_rate * client_scale, label = percent(win_rate, accuracy = 0.1)),
    nudge_y = 1700,
    size = 3.0,
    color = "#0F172A",
    fontface = "bold"
  ) +
  scale_y_continuous(
    name = "Opportunity count",
    labels = comma,
    sec.axis = sec_axis(~ . / client_scale, name = "Observed win rate", labels = percent_format(accuracy = 1))
  ) +
  labs(x = NULL, title = NULL, subtitle = NULL) +
  theme_slide +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    axis.line.y.left = element_line(color = "#CBD5E1"),
    axis.line.y.right = element_line(color = "#CBD5E1"),
    axis.line.x = element_line(color = "#CBD5E1")
  )

ggsave(client_asset, client_plot, width = 8.6, height = 4.8, dpi = 220, bg = "white")

funnel_df <- tibble::tibble(
  phase = factor(c("(Re)Acquisition", "Engagement / upselling"), levels = c("(Re)Acquisition", "Engagement / upselling")),
  opportunities = c(69208, 8817),
  win_rate = c(0.173, 0.639)
)

max_funnel_count <- max(funnel_df$opportunities)
funnel_scale <- max_funnel_count / 0.75

funnel_plot <- ggplot(funnel_df, aes(x = phase)) +
  geom_col(aes(y = opportunities), fill = "#93C5FD", width = 0.62) +
  geom_text(
    aes(y = opportunities, label = comma(opportunities)),
    vjust = -0.45,
    size = 3.4,
    color = "#1E3A8A",
    fontface = "bold"
  ) +
  geom_line(
    aes(y = win_rate * funnel_scale, group = 1),
    color = "#0F172A",
    linewidth = 1.3
  ) +
  geom_point(aes(y = win_rate * funnel_scale), color = "#0F172A", size = 2.8) +
  geom_text(
    aes(y = win_rate * funnel_scale, label = percent(win_rate, accuracy = 0.1)),
    nudge_y = c(2500, 2500),
    size = 3.2,
    color = "#0F172A",
    fontface = "bold"
  ) +
  scale_y_continuous(
    name = "Opportunity count",
    labels = comma,
    sec.axis = sec_axis(~ . / funnel_scale, name = "Observed win rate", labels = percent_format(accuracy = 1))
  ) +
  labs(x = NULL, title = NULL, subtitle = NULL) +
  theme_slide +
  theme(
    axis.text.x = element_text(size = 10),
    axis.line.y.left = element_line(color = "#CBD5E1"),
    axis.line.y.right = element_line(color = "#CBD5E1"),
    axis.line.x = element_line(color = "#CBD5E1")
  )

ggsave(funnel_asset, funnel_plot, width = 8.3, height = 4.8, dpi = 220, bg = "white")

cat(sprintf("saved: %s\n", client_asset))
cat(sprintf("saved: %s\n", funnel_asset))
