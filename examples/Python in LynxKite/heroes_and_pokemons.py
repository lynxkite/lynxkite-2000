import pandas as pd
from datasets import load_dataset
from rapidfuzz import process, fuzz
import io
import matplotlib.pyplot as plt
import requests


def load_superheroes():
    """Load Superheroes dataset"""
    heroes_ds = load_dataset("jrtec/Superheroes", split=["train", "test"])
    heroes_df = pd.concat([ds.to_pandas() for ds in heroes_ds])
    return heroes_df


def load_pokemon():
    type_names_csv = pd.read_csv(
        "https://raw.githubusercontent.com/veekun/pokedex/refs/heads/master/pokedex/data/csv/types.csv"
    )
    pokemon_names_csv = pd.read_csv(
        "https://raw.githubusercontent.com/veekun/pokedex/refs/heads/master/pokedex/data/csv/pokemon.csv"
    )
    pokemon_types_csv = pd.read_csv(
        "https://raw.githubusercontent.com/veekun/pokedex/refs/heads/master/pokedex/data/csv/pokemon_types.csv"
    )

    pokemon_types_list = (
        pokemon_types_csv.merge(
            type_names_csv[["id", "identifier"]],
            left_on="type_id",
            right_on="id",
            suffixes=("", "_type"),
        )
        .groupby("pokemon_id")["identifier"]
        .apply(list)
        .reset_index()
    )
    # Get the list of types for each Pokemon. A Pokemon can have multiple types. Keep Pokemon name and ID.
    pokemon_df = pokemon_types_list.merge(
        pokemon_names_csv[["id", "identifier"]], left_on="pokemon_id", right_on="id"
    )
    pokemon_df = pokemon_df.rename(columns={"identifier_y": "name", "identifier_x": "types"})
    return pokemon_df


def capitalize_field(df, field_name):
    df = df.copy()
    df[field_name] = df[field_name].str.capitalize()
    return df


def match_pokemon(heroes_df, pokemon_df):
    pokemon_names = pokemon_df["name"].astype(str).tolist()

    def match_pokemon_for(hero_name, pokemon_names, scorer=fuzz.WRatio):
        """
        Returns best matching Pokémon name and similarity score.
        """
        match, score, _ = process.extractOne(hero_name, pokemon_names, scorer=scorer)
        return match, score

    matches = heroes_df["name"].apply(lambda x: match_pokemon_for(x, pokemon_names))
    heroes_df = heroes_df.copy()
    heroes_df["matched_pokemon"] = matches.apply(lambda x: x[0])
    pokemon_name_to_id = pokemon_df.set_index("name")["pokemon_id"].to_dict()
    heroes_df["matched_pokemon_id"] = (
        heroes_df["matched_pokemon"].str.lower().map(pokemon_name_to_id)
    )
    heroes_df["pokemon_similarity"] = matches.apply(lambda x: x[1])
    heroes_df = heroes_df.sort_values(by="pokemon_similarity", ascending=False)
    return heroes_df


def plot_match_histogram(heroes_df):
    heroes_df.pokemon_similarity.plot.hist(bins=20)


def plot_top_matches(heroes_df, n):
    # Top n matches
    top = heroes_df.nlargest(n, "pokemon_similarity")[
        ["name", "matched_pokemon", "matched_pokemon_id", "pokemon_similarity"]
    ]

    # Create image URLs
    def pokemon_image_url(pokemon_id):
        pokemon_id = str(pokemon_id).zfill(3)
        return f"https://raw.githubusercontent.com/HybridShivam/Pokemon/refs/heads/master/assets/images/{pokemon_id}.png"

    top["image_url"] = top["matched_pokemon_id"].apply(pokemon_image_url)

    # --- Plot ---
    height = 0.9
    top["label"] = top["name"] + " → " + top["matched_pokemon"]
    plt.barh(
        y=top.label, width=top.pokemon_similarity, height=height, color="skyblue", align="center"
    )
    plt.title(f"Top {n} Superhero ↔ Pokémon Matches", fontsize=14)
    plt.xlim(0, 100)
    plt.ylim(-1, len(top))

    for i, row in enumerate(top.to_records()):
        response = requests.get(row.image_url)
        img = plt.imread(io.BytesIO(response.content))
        w = row.pokemon_similarity
        plt.imshow(
            img, extent=[w - 12, w - 2, i - height / 2, i + height / 2], aspect="auto", zorder=2
        )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def main():
    heroes_df = load_superheroes()
    pokemon_df = load_pokemon()
    pokemon_df = capitalize_field(pokemon_df, "name")
    heroes_df = match_pokemon(heroes_df, pokemon_df)
    plot_match_histogram(heroes_df)
    plot_top_matches(heroes_df, n=8)
