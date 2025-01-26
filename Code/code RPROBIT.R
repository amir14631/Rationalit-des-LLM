# Charger la bibliothèque RprobitB
library(RprobitB)
data <- read.csv("C:/Users/amirb/Downloads/human_vs_ia_choice.csv")

# Définir la formule pour analyser `ai_choice`
form <- ai_choice_mistral ~ price + time + change + comfort | 0
data <- prepare_data(form, data, id = "deciderID", idc = "occasionID")
model <- fit_model(data, scale = "price := -1")


# Afficher les résultats
plot(coef(model))


predict(
  model,
  data = data.frame(
    "price_A" = c(100, 110),
    "price_B" = c(100, 100)
  ),
  overview = FALSE
)



predict(
  model,
  data = data.frame(
    "price_A" = c(100, 110),
    "comfort_A" = c(1, 0),
    "price_B" = c(100, 100),
    "comfort_B" = c(1, 1)
  ),
  overview = FALSE
)


predict(
  model,
  data = data.frame(
    "price_A" = c(100, 110),
    "change_A" = c(1, 0),
    "price_B" = c(100, 100),
    "change_B" = c(1, 1)
  ),
  overview = FALSE
)





predict(
  model,
  data = data.frame(
    "price_A" = c(100, 110),
    "time_A" = c(2.5, 1.5),
    "price_B" = c(100, 100),
    "time_B" = c(2.5, 2.5)
  ),
  overview = FALSE
)

