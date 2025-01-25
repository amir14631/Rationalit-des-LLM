# Rationalité-des-LLM
Ce projet analyse la rationalité des IA (Gemini, ChatGPT, Mistral) en comparant leurs choix à 2929 décisions humaines dans un scénario de billets (prix, temps, correspondances, confort). Avec RprobitB, il explore leur potentiel en études de marché pour modéliser les préférences économiques.

# Méthodologie
### Dataset
Le dataset original, téléchargé depuis le repository GitHub de Loël Schlaeger (https://github.com/loelschlaeger/RprobitB/blob/main/data/train_choice.rda), contient des données de choix humains dans un scénario de sélection entre deux options de billets de train (A et B).
Résumé des Variables

    Prix (price_A, price_B) : Coût du billet pour chaque option (en euros).
    Durée du trajet (time_A, time_B) : Durée totale du voyage (en heures).
    Changements nécessaires (change_A, change_B) : Nombre de correspondances pour effectuer le trajet.
    Niveau de confort (comfort_A, comfort_B) : (mesure inversée) Évalué sur une échelle de 2 à 0, où 0 représente le confort maximal et 2 le confort minimal.

Le dataset, initialement au format .rda, a été transformé en fichier CSV pour faciliter son utilisation dans des scripts Python. Cette transformation permet d’intégrer les données dans des workflows d’analyse et de simulation avec des modèles d’intelligence artificielle.


---

### **Fonctionnement des API et Explication des Paramètres**

Dans cette étude, trois modèles d’intelligence artificielle (**ChatGPT**, **Gemini**, et **Mistral**) ont été utilisés pour simuler des choix humains entre deux options de billets (**A** et **B**). Chaque IA a reçu les variables des billets (prix, temps de trajet, changements, confort) sous forme de prompts clairs, et leurs réponses ont été enregistrées pour une analyse comparative avec les choix humains.

---

### **Prompt Commun**
Pour les trois API, le même **prompt** a été utilisé pour garantir une cohérence dans la façon dont les données sont présentées aux modèles. Chaque prompt inclut les critères suivants pour contextualiser la tâche :

```python
"Tu es un passager de train et tu dois choisir entre deux options pour ton voyage. Voici les critères importants :
1. Prix : Le coût du billet en euros. Un prix plus bas est préféré.
2. Temps : La durée totale du trajet en heures. Un trajet plus court est préféré.
3. Changements : Le nombre de correspondances pendant le trajet. Moins de changements sont préférés.
4. Confort : Une mesure inversée du confort (plus bas = plus confortable)."
```

Le contenu du prompt, propre à chaque ligne de données, est ensuite intégré comme suit :
```python
prompt = f"""
Voici deux options :
- Option A : Prix = {row['price_A']} €, Temps = {row['time_A']} heures, Changements = {row['change_A']}, Confort = {row['comfort_A']}
- Option B : Prix = {row['price_B']} €, Temps = {row['time_B']} heures, Changements = {row['change_B']}, Confort = {row['comfort_B']}
Basé uniquement sur ces informations selon tes critères, quelle option choisirais-tu ? Répondez par 'A' ou 'B' seulement.
"""
```

---

### **Choix de la Température (0.7)**

La température est un paramètre essentiel qui contrôle la **variabilité des réponses** d’une IA. Après plusieurs tests avec différentes valeurs, le choix de **0.7** a été retenu pour les raisons suivantes :

- **Température 0 :** Produisait des réponses trop rigides et déterministes, limitant l’adaptabilité des modèles.
- **Température 1 :** Menait à des hallucinations fréquentes (comme produire des phrases complètes ou ignorer les consignes de répondre uniquement par **A** ou **B**).
- **Température 0.7 :** Offrait un compromis idéal entre diversité et cohérence :
  - Réduisait les hallucinations.
  - Garantissait que les modèles respectaient mieux les consignes.
  - Bien que certains modèles, comme **Mistral**, puissent mieux performer à **1**, le choix de **0.7** a permis d’uniformiser les résultats entre tous les modèles pour une meilleure cohérence globale.

---
### **Pourquoi traiter par batch ?**

Le traitement par batch permet de :
1. **Respecter les limites des API** (ex. 1 requête/sec pour Mistral, 15 req/min pour Gemini) en ajoutant des pauses.
2. **Gérer les erreurs** en identifiant et isolant les lignes problématiques sans interrompre tout le processus.
3. **Sauvegarder les résultats progressivement** pour éviter la perte de données en cas d'interruption.
4. **Optimiser les performances** en assurant un traitement fluide sans surcharger les API.

### **Explication de Max Tokens**

Le paramètre **Max Tokens** définit la longueur maximale de la réponse générée par l'IA. Dans ce projet, il est fixé à **10 tokens** , meme si theoriquement il pourrais ete fixé a **1 token** car le prompte limite les réponses à une seule lettre (**A** ou **B**).
## **API Mistral**

#### **Caractéristiques**
- **Modèle utilisé :** `mistral-small-latest`  
- **Paramètres :**
  - **Température :** 0.7 
- **Particularités :**
  - **API gratuite**
  - Limite : **1 requête par seconde**

#### **Code Complet**
```python
import requests
import pandas as pd
import time

api_url = "https://api.mistral.ai/v1/chat/completions"
api_key = "Votre_clé_API"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

file_path = r"C: le chemin de train_choice.csv"
dataset = pd.read_csv(file_path)

def get_ai_decision(row):
    prompt = f"""
    Tu es un passager de train et tu dois choisir entre deux options pour ton voyage. Voici les critères importants :
    1. Prix : Le coût du billet en euros. Un prix plus bas est préféré.
    2. Temps : La durée totale du trajet en heures. Un trajet plus court est préféré.
    3. Changements : Le nombre de correspondances pendant le trajet. Moins de changements sont préférés.
    4. Confort : Une mesure inversée du confort (plus bas = plus confortable).

    Voici deux options :
    - Option A : Prix = {row['price_A']} €, Temps = {row['time_A']} heures, Changements = {row['change_A']}, Confort = {row['comfort_A']}
    - Option B : Prix = {row['price_B']} €, Temps = {row['time_B']} heures, Changements = {row['change_B']}, Confort = {row['comfort_B']}

    Basé uniquement sur ces informations selon tes critères, quelle option choisirais-tu ? Répondez par 'A' ou 'B' seulement.
    """
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": "Tu es un assistant qui aide à prendre des décisions basées sur des informations simples."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,
        "temperature": 0.7
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Erreur avec l'API pour la ligne {row.name} : {e}")
        return "Erreur"

output_path = r"C:\Users\mistral_choice.csv" # a modifier selon le fichier d'output voulu

with open(output_path, "w") as f:
    f.write(",".join(dataset.columns) + ",ai_choice\n")

for index, row in dataset.iterrows():
    ai_choice = get_ai_decision(row)
    dataset.loc[index, "ai_choice"] = ai_choice
    with open(output_path, "a") as f:
        f.write(",".join(map(str, row.tolist())) + f",{ai_choice}\n")
    print(f"Ligne {index + 1}/{len(dataset)} traitée : {ai_choice}")
    time.sleep(2)

print(f"Traitement terminé. Fichier final sauvegardé dans : {output_path}")
```

---

### **API ChatGPT**

#### **Caractéristiques**
- **Modèle utilisé :** `gpt-4o`  
- **Paramètres :**
  - **Température :** 0.7
- **Particularités :**
  - **API payante**
  - Coût total estimé : **1.80 €** pour traiter 2929 lignes

#### **Code Complet**
```python
from openai import OpenAI
import pandas as pd
import time

client = OpenAI(api_key="Votre_clé_API")

file_path = r"C: le chemin de train_choice.csv"
dataset = pd.read_csv(file_path)

def get_ai_decision(row):
    prompt = f"""
    Tu es un passager de train et tu dois choisir entre deux options pour ton voyage. Voici les critères importants :
    1. Prix : Le coût du billet en euros. Un prix plus bas est préféré.
    2. Temps : La durée totale du trajet en heures. Un trajet plus court est préféré.
    3. Changements : Le nombre de correspondances pendant le trajet. Moins de changements sont préférés.
    4. Confort : Une mesure inversée du confort (plus bas = plus confortable).

    Voici deux options :
    - Option A : Prix = {row['price_A']} €, Temps = {row['time_A']} heures, Changements = {row['change_A']}, Confort = {row['comfort_A']}
    - Option B : Prix = {row['price_B']} €, Temps = {row['time_B']} heures, Changements = {row['change_B']}, Confort = {row['comfort_B']}

    En tant que passager, quelle option choisirais-tu ? Réponds par 'A' ou 'B' seulement.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un assistant qui aide à prendre des décisions basées sur des informations simples"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erreur avec l'API pour la ligne {row.name} : {e}")
        return "Erreur"

output_path = r"C:\Users\amirb\Downloads\train_chatgpt.csv"

for index, row in dataset.iterrows():
    ai_choice = get_ai_decision(row)
    dataset.loc[index, "ai_choice_gpt"] = ai_choice
    print(f"Choix enregistré pour la ligne {index + 1}/{len(dataset)} : {ai_choice}")
    time.sleep(1)

dataset.to_csv(output_path, index=False)
print(f"Traitement terminé. Fichier final sauvegardé dans : {output_path}")
```

---

### **API Gemini**

#### **Caractéristiques**
- **Modèle utilisé :** `gemini-1.5-flash`  
- **Paramètres :**
  - **Température :** 0.7  
- **Particularités :**
  - Limite : **15 requêtes par minute**  
  - Limite : **1 000 000 tokens par minute**  
  - Limite : **1 500 requêtes par jour**  

#### **Code Complet**
```python
import time
import google.generativeai as genai
import pandas as pd

# Configurer l'API Gemini
api_key = "Votre_clé_API"  # Remplace par ta clé
genai.configure(api_key=api_key)

# Charger les données d'origine
file_path = r"C: le chemin de train_choice.csv"
dataset = pd.read_csv(file_path)

# Fonction pour interroger l'API Gemini
def get_ai_decision(row):
    prompt = f"""
    Tu es un passager de train et tu dois choisir entre deux options pour ton voyage. Voici les critères importants :
    1. Prix : Le coût du billet en euros. Un prix plus bas est préféré.
    2. Temps : La durée totale du trajet en heures. Un trajet plus court est préféré.
    3. Changements : Le nombre de correspondances pendant le trajet. Moins de changements sont préférés.
    4. Confort : Une mesure inversée du confort (plus bas = plus confortable).

    Voici deux options disponibles :
    - Option A : Prix = {row['price_A']} €, Temps = {row['time_A']} heures, Changements = {row['change_A']}, Confort = {row['comfort_A']}
    - Option B : Prix = {row['price_B']} €, Temps = {row['time_B']} heures, Changements = {row['change_B']}, Confort = {row['comfort_B']}

    En tant que passager, quelle option choisirais-tu ? Réponds par 'A' ou 'B' seulement.
    """
    try:
        # Utiliser le modèle Gemini pour générer une réponse
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 10,
            },
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Erreur avec l'API pour la ligne {row.name} : {e}")
        return "Erreur"

# Sauvegarde ligne par ligne dans le fichier train_gemini.csv
output_path = r"C:train_gemini.csv"
with open(output_path, "w") as f:
    # Écrire les en-têtes
    f.write(",".join(dataset.columns) + ",ai_choice\n")

# Traiter ligne par ligne avec respect des limites
for index, row in dataset.iterrows():
    ai_choice = get_ai_decision(row)
    dataset.loc[index, 'ai_choice'] = ai_choice
    # Sauvegarder la ligne dans le fichier
    with open(output_path, "a") as f:
        f.write(",".join(map(str, row.tolist())) + f",{ai_choice}\n")
    print(f"Ligne {index + 1}/{len(dataset)} traitée : {ai_choice}")
    
    # Pause pour respecter les limites de 15 requêtes par minute
    time.sleep(4)  # Pause de 4 secondes

print(f"Traitement terminé. Fichier final sauvegardé dans : {output_path}")
```
### **Résultats**

#### **Fusion des datasets générés**
Avant d’analyser les résultats, il est recommandé de fusionner les trois datasets générés à partir des API (**ChatGPT**, **Gemini**, et **Mistral**) en un seul dataset. Cela permet de faciliter les comparaisons et de calculer directement les correspondances entre les choix humains et ceux des modèles IA.

Le fichier fusionné, nommé **`train_human_vs_ia_merged.csv`**, contient :
- Les choix humains (**choice**) pour chaque ligne.
- Les choix simulés par chaque IA (**ai_choice_gpt**, **ai_choice_mistral**, **ai_choice_gemini**).

---

#### **Code pour calculer les correspondances**
Le code suivant calcule le pourcentage de correspondance entre les choix humains et ceux de chaque modèle IA :

```python
import pandas as pd

# Charger le dataset fusionné
file_path = r"C:\Users\amirb\Downloads\train_human_vs_ia_merged.csv"
dataset = pd.read_csv(file_path)

# Calculer les correspondances
for model in ['ai_choice_gpt', 'ai_choice_mistral', 'ai_choice_gemini']:
    match_percentage = (dataset['choice'] == dataset[model]).mean() * 100
    print(f"Pourcentage de correspondance entre Humain et {model} : {match_percentage:.2f}%")
```

---

#### **Résultats obtenus**
Voici les résultats calculés à partir des données :

- **Correspondance entre Humains et ai_choice_gemini :** **65.59 %**
- **Correspondance entre Humains et ai_choice_gpt :** **65.28 %**
- **Correspondance entre Humains et ai_choice_mistral :** **62.34 %**

---

### **Analyse des variables choisies**

#### **Objectif**
Ce code analyse la distribution des valeurs des variables sélectionnées (**price**, **time**, **change**, et **comfort**) par les humains et les modèles IA (**ChatGPT**, **Gemini**, et **Mistral**). L’objectif est de comparer comment chaque groupe (humains et modèles) priorise ces variables lors de la prise de décision.

---

#### **Code pour l'analyse des distributions**

Voici le code utilisé pour analyser et visualiser les distributions des valeurs choisies :

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger le dataset fusionné
file_path = r"C:\Users\amirb\Downloads\train_human_vs_ia_merged.csv"
dataset = pd.read_csv(file_path)

# Liste des variables et modèles à analyser
variables = ['price', 'time', 'change', 'comfort']
models = ['choice', 'ai_choice_gpt', 'ai_choice_mistral', 'ai_choice-gemini']

# Ajouter des colonnes pour les variables choisies
for var in variables:
    for model in models:
        column_name = f"chosen_{var}_{model}"
        dataset[column_name] = dataset.apply(
            lambda row: row[f"{var}_A"] if row[model] == 'A' else row[f"{var}_B"],
            axis=1
        )

# Visualisation des distributions
for var in variables:
    plt.figure(figsize=(10, 6))
    for model in models:
        column_name = f"chosen_{var}_{model}"
        dataset[column_name].hist(alpha=0.5, label=model, bins=20)
    plt.title(f"Distribution de {var} choisi par les humains et les modèles")
    plt.xlabel(var.capitalize())


    plt.ylabel("Fréquence")
    plt.legend()
    plt.show()
```

---
### **Méthode Probit avec {RprobitB}**

*Ce texte est une traduction en français de la page officielle disponible sur [le dépôt GitHub de {RprobitB}](https://github.com/loelschlaeger/RprobitB/blob/main/README.md).*

---

#### **Objectif de {RprobitB}**
Le package **{RprobitB}** est conçu pour expliquer les choix effectués par des décideurs parmi un ensemble discret d'alternatives, en utilisant une approche bayésienne. Par exemple, il peut être utilisé pour comprendre pourquoi des touristes préfèrent un billet de train plutôt qu’un autre. Ces informations sont précieuses pour les entreprises ferroviaires, notamment pour évaluer la **disposition à payer** des clients pour des trajets plus rapides ou plus confortables.

---

#### **Installation**
Le package peut être installé directement depuis **CRAN** avec la commande suivante :

```r
install.packages("RprobitB")
```

---

#### **Documentation**
La documentation complète et les vignettes sont disponibles sur le site officiel du package : [Documentation RprobitB](https://loelschlaeger.de/RprobitB/).

---

#### **Exemple d’utilisation**
Dans cette étude, **{RprobitB}** a été utilisé pour analyser un dataset contenant **2929 choix** effectués par **235 individus néerlandais** confrontés à deux options de voyage en train. Les alternatives sont décrites par les variables suivantes :
- **Prix** (en euros)
- **Temps de trajet** (en heures)
- **Confort** (plus faible = plus confortable)
- **Nombre de correspondances**

---

#### **Ajustement du modèle Probit**
Un modèle probit est ajusté pour expliquer les choix en fonction des variables mentionnées. La spécification du modèle est donnée par la formule suivante :

```r
form <- choice ~ price + time + change + comfort | 0
```

Dans cette formule :
- **`choice`** est la variable cible (le choix effectué par les individus).
- **`price`, `time`, `change`, et `comfort`** sont les variables explicatives.

Le coefficient du **prix** est fixé à **-1** pour permettre une interprétation monétaire des autres coefficients.

##### **Code pour ajuster le modèle**
```r
data <- prepare_data(form, train_choice, id = "deciderID", idc = "occasionID")
model <- fit_model(data, scale = "price := -1")
```

---

#### **Résultats estimés**
Les coefficients estimés montrent la valorisation des individus pour chaque variable en termes monétaires :
- **Une heure de trajet supplémentaire** : ~25 €
- **Une correspondance supplémentaire** : ~5 €
- **Un niveau de confort amélioré** : ~15 €

Ces valeurs permettent d’évaluer la disposition à payer des utilisateurs pour des améliorations spécifiques des services.

##### **Visualisation des coefficients**
Les coefficients peuvent être visualisés pour une meilleure interprétation :

```r
plot(coef(model))
```

---

#### **Simulations de parts de marché**
Grâce au modèle probit ajusté, il est possible de simuler l’effet de changements spécifiques sur les parts de marché.

1. **Augmentation du prix**
   Une augmentation du prix de **100 € à 110 €** (toutes choses égales par ailleurs) entraîne une perte de **15 %** des clients au profit de l’option concurrente.

   ```r
   predict(
     model,
     data = data.frame(
       "price_A" = c(100, 110),
       "price_B" = c(100, 100)
     ),
     overview = FALSE
   )
   ```

2. **Amélioration du confort**
   En améliorant le confort de la classe pour compenser une augmentation de prix, on observe un **gain de 7 %** des parts de marché.

   ```r
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
   ```

---

#### **Application dans notre étude**
Dans ce projet, la méthode probit a été utilisée pour analyser et comparer les choix humains et les décisions générées par les modèles IA (**ChatGPT**, **Gemini**, et **Mistral**). L’objectif est d’évaluer la capacité des IA à reproduire les priorités humaines et de comprendre les différences dans la valorisation des variables.

Souhaitez-vous d'autres précisions ou des ajustements pour cette section ? 😊
