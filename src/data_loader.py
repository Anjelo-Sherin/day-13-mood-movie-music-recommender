"""Data loader and dataset initialization for Mood-Based Movie and Music Recommender."""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def generate_default_datasets():
    """Generates rich, authentic MovieLens and Spotify datasets if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    spotify_path = os.path.join(DATA_DIR, "spotify_tracks.csv")

    if not os.path.exists(movies_path):
        movies = [
            # High Valence, High Energy (Joyful, Exciting, Fun, Uplifting)
            (1, "Spider-Man: Into the Spider-Verse", 2018, "Animation|Action|Adventure", 
             "Teen Miles Morales becomes the new Spider-Man, joining forces with five alternate universe counterparts to stop a threat to all reality.",
             8.4, 21500, "Bob Persichetti", 0.85, 0.82, "uplifting|energetic|heroic|vibrant",
             "https://images.unsplash.com/photo-1635805737707-575885ab0820?w=500&auto=format&fit=crop&q=80"),
            (2, "Singin' in the Rain", 1952, "Comedy|Musical|Romance",
             "A silent film star falls for a chorus girl just as he and his delusional screen partner attempt the difficult transition to talking pictures.",
             8.3, 14200, "Stanley Donen", 0.92, 0.78, "joyful|euphoric|classic|romantic",
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80"),
            (3, "The Grand Budapest Hotel", 2014, "Adventure|Comedy|Crime",
             "A writer encounters the owner of an aging high-class European hotel, who tells him of his early years serving as a lobby boy in the glorious glory years.",
             8.1, 35000, "Wes Anderson", 0.82, 0.75, "quirky|whimsical|witty|uplifting",
             "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=500&auto=format&fit=crop&q=80"),
            (4, "School of Rock", 2003, "Comedy|Music",
             "After being kicked out of his rock band, Dewey Finn becomes a substitute teacher at a strict elementary private school and turns his class into a rock band.",
             7.2, 19800, "Richard Linklater", 0.88, 0.85, "rocking|fun|inspiring|energetic",
             "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=80"),
            (5, "Guardians of the Galaxy", 2014, "Action|Adventure|Comedy",
             "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe, powered by an awesome mixtape.",
             8.0, 42000, "James Gunn", 0.84, 0.88, "funny|hype|badass|cosmic",
             "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?w=500&auto=format&fit=crop&q=80"),
            (6, "Everything Everywhere All at Once", 2022, "Action|Adventure|Comedy|Sci-Fi",
             "A middle-aged Chinese immigrant is swept into an insane adventure in which she alone can save existence by exploring alternate universes connecting to the lives she could have led.",
             7.8, 38000, "Daniels", 0.75, 0.92, "mind-bending|absurdist|heartfelt|exhilarating",
             "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=80"),
            (7, "Superbad", 2007, "Comedy",
             "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry.",
             7.6, 26000, "Greg Mottola", 0.86, 0.80, "hilarious|nostalgic|chaotic|upbeat",
             "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80"),
            (8, "Ferris Bueller's Day Off", 1986, "Comedy",
             "A high school wise guy is determined to have a day off from school, despite what the Principal thinks.",
             7.8, 22000, "John Hughes", 0.90, 0.79, "carefree|euphoric|fun|classic",
             "https://images.unsplash.com/photo-1509281373149-e957c6296406?w=500&auto=format&fit=crop&q=80"),

            # High Valence, Low Energy (Peaceful, Warm, Comforting, Calming)
            (9, "Amélie", 2001, "Comedy|Romance",
             "Amélie is an innocent and naive girl in Paris with her own sense of justice. She decides to help those around her and, along the way, discovers love.",
             8.3, 31000, "Jean-Pierre Jeunet", 0.86, 0.42, "wholesome|warm|enchanting|comforting",
             "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=500&auto=format&fit=crop&q=80"),
            (10, "Paddington 2", 2017, "Adventure|Comedy|Family",
             "Paddington, now happily settled with the Brown family, picks up a series of odd jobs to buy the perfect present for his Aunt Lucy's 100th birthday.",
             7.8, 16000, "Paul King", 0.94, 0.45, "wholesome|comfort|gentle|sweet",
             "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=80"),
            (11, "My Neighbor Totoro", 1988, "Animation|Comedy|Family|Fantasy",
             "When two girls move to the country to be near their ailing mother, they have adventures with the wondrous forest spirits who live nearby.",
             8.1, 28000, "Hayao Miyazaki", 0.89, 0.38, "pastoral|soothing|nostalgic|peaceful",
             "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=500&auto=format&fit=crop&q=80"),
            (12, "Before Sunrise", 1995, "Drama|Romance",
             "A young man and an American woman meet on a train in Europe and wind up spending one evening together in Vienna before parting ways.",
             8.1, 24000, "Richard Linklater", 0.81, 0.35, "intimate|conversational|dreamy|romantic",
             "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=500&auto=format&fit=crop&q=80"),
            (13, "About Time", 2013, "Comedy|Drama|Fantasy|Romance",
             "At the age of 21, Tim discovers he can travel in time and change what happens and has happened in his own life, focusing on cherishing each day.",
             7.8, 27000, "Richard Curtis", 0.83, 0.48, "heartwarming|grateful|tender|reflective",
             "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=500&auto=format&fit=crop&q=80"),
            (14, "Soul", 2020, "Animation|Adventure|Comedy|Drama",
             "After landing the gig of a lifetime, a New York jazz pianist suddenly finds himself trapped in a strange land between Earth and the afterlife.",
             8.0, 32000, "Pete Docter", 0.82, 0.46, "philosophical|cozy|inspiring|healing",
             "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=500&auto=format&fit=crop&q=80"),
            (15, "Midnight in Paris", 2011, "Comedy|Fantasy|Romance",
             "While on a trip to Paris with his fiancée's family, a nostalgic Hollywood screenwriter finds himself mysteriously going back to the 1920s every night.",
             7.7, 23000, "Woody Allen", 0.80, 0.40, "nostalgic|cozy|artsy|dreamy",
             "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=500&auto=format&fit=crop&q=80"),
            (16, "Little Miss Sunshine", 2006, "Comedy|Drama",
             "A family determined to get their young daughter into the finals of a beauty pageant take a cross-country trip in their VW bus.",
             7.8, 25000, "Jonathan Dayton", 0.80, 0.52, "uplifting|dysfunctional|sweet|funny",
             "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=500&auto=format&fit=crop&q=80"),

            # Low Valence, Low Energy (Melancholic, Sad, Reflective, Thoughtful)
            (17, "Eternal Sunshine of the Spotless Mind", 2004, "Drama|Romance|Sci-Fi",
             "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories.",
             8.3, 44000, "Michel Gondry", 0.32, 0.44, "melancholic|heartbreaking|poetic|bittersweet",
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80"),
            (18, "Her", 2013, "Drama|Romance|Sci-Fi",
             "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need.",
             8.0, 33000, "Spike Jonze", 0.42, 0.32, "lonely|tender|melancholy|contemplative",
             "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500&auto=format&fit=crop&q=80"),
            (19, "Past Lives", 2023, "Drama|Romance",
             "Nora and Hae Sung, two deeply connected childhood friends, are wrest apart after Nora's family emigrates from South Korea. Two decades later, they reunite in New York.",
             7.9, 18000, "Celine Song", 0.38, 0.28, "wistful|pensive|longing|gentle",
             "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=500&auto=format&fit=crop&q=80"),
            (20, "Manchester by the Sea", 2016, "Drama",
             "A depressed uncle is asked to take care of his teenage nephew after the boy's father dies.",
             7.8, 24000, "Kenneth Lonergan", 0.20, 0.30, "grief|somber|realistic|cathartic",
             "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=500&auto=format&fit=crop&q=80"),
            (21, "Lost in Translation", 2003, "Comedy|Drama",
             "A faded movie star and a neglected young woman form an unlikely bond after crossing paths in Tokyo.",
             7.7, 26000, "Sofia Coppola", 0.45, 0.30, "alienation|tokyo|atmospheric|wistful",
             "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?w=500&auto=format&fit=crop&q=80"),
            (22, "Moonlight", 2016, "Drama",
             "A young African-American man grapples with his identity and sexuality while experiencing the everyday struggles of childhood, adolescence, and burgeoning adulthood.",
             7.4, 21000, "Barry Jenkins", 0.35, 0.34, "intimate|lyrical|poignant|honest",
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80"),
            (23, "The Shawshank Redemption", 1994, "Drama",
             "Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion.",
             9.3, 75000, "Frank Darabont", 0.58, 0.48, "endurance|hope|inspiring|profound",
             "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=80"),
            (24, "Blade Runner 2049", 2017, "Action|Drama|Mystery|Sci-Fi",
             "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who's been missing for thirty years.",
             8.0, 39000, "Denis Villeneuve", 0.35, 0.50, "existential|cyberpunk|brooding|hypnotic",
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80"),

            # Low Valence, High Energy (Intense, Dark, Thrilling, Angry, Gritty)
            (25, "Whiplash", 2014, "Drama|Music",
             "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
             8.5, 46000, "Damien Chazelle", 0.40, 0.95, "intense|relentless|obsessive|adrenaline",
             "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=80"),
            (26, "The Dark Knight", 2008, "Action|Crime|Drama",
             "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
             9.0, 82000, "Christopher Nolan", 0.35, 0.92, "gritty|chaotic|electrifying|tense",
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80"),
            (27, "Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi",
             "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners, a psychotic worshiper and an ex-drifter named Max.",
             8.1, 49000, "George Miller", 0.45, 0.98, "furious|kinetic|wild|adrenaline",
             "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=80"),
            (28, "Parasite", 2019, "Drama|Thriller",
             "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
             8.5, 48000, "Bong Joon Ho", 0.38, 0.76, "suspenseful|dark comedy|tense|masterpiece",
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80"),
            (29, "Fight Club", 1999, "Drama",
             "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
             8.8, 65000, "David Fincher", 0.30, 0.88, "anarchic|rebellious|raw|dark",
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80"),
            (30, "Pulp Fiction", 1994, "Crime|Drama",
             "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
             8.9, 71000, "Quentin Tarantino", 0.55, 0.84, "cool|stylish|sharp|dialogue",
             "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=80"),
            (31, "Interstellar", 2014, "Adventure|Drama|Sci-Fi",
             "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft along with a team of researchers to find a new planet for humans.",
             8.7, 68000, "Christopher Nolan", 0.52, 0.78, "epic|emotional|cosmic|thrilling",
             "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=500&auto=format&fit=crop&q=80"),
            (32, "Inception", 2010, "Action|Adventure|Sci-Fi",
             "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
             8.8, 79000, "Christopher Nolan", 0.48, 0.86, "mind-bending|pacing|complex|cinematic",
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80")
        ]

        df_movies = pd.DataFrame(movies, columns=[
            "movie_id", "title", "release_year", "genres", "overview",
            "vote_average", "vote_count", "director", "mood_valence", "mood_energy",
            "mood_tags", "poster_url"
        ])
        df_movies.to_csv(movies_path, index=False)

    if not os.path.exists(ratings_path):
        # Generate representative user ratings matrix for SVD collaborative filtering
        np.random.seed(42)
        records = []
        n_users = 40
        n_movies = 32
        
        # User personas:
        # Users 1-10: Love uplifting/comedy/whimsical
        # Users 11-20: Love dark/action/intense
        # Users 21-30: Love melancholic/indie/thoughtful
        # Users 31-40: Broad cinematic/classic enthusiasts
        
        for u in range(1, n_users + 1):
            for m in range(1, n_movies + 1):
                # Sample probability of having rated the movie (density ~40%)
                if np.random.rand() < 0.45:
                    if u <= 10:
                        # Uplifting bias: high for movies 1-16
                        base = 4.5 if m <= 16 else 2.5
                    elif u <= 20:
                        # Intense/dark bias: high for movies 25-32
                        base = 4.6 if m >= 25 else 2.8
                    elif u <= 30:
                        # Melancholic/indie bias: high for 9-24
                        base = 4.6 if (9 <= m <= 24) else 3.0
                    else:
                        base = 3.8
                    rating = np.clip(np.round(base + np.random.normal(0, 0.6) * 2) / 2, 1.0, 5.0)
                    records.append((u, m, rating, 1600000000 + np.random.randint(0, 50000000)))

        df_ratings = pd.DataFrame(records, columns=["userId", "movieId", "rating", "timestamp"])
        df_ratings.to_csv(ratings_path, index=False)

    if not os.path.exists(spotify_path):
        tracks = [
            # High Valence, High Energy (Party, Euphoric, Happy, Upbeat)
            ("track_01", "Don't Stop Me Now", "Queen", "Jazz", "rock", 0.96, 0.87, 156.0, 0.56, 0.05, 88, 209000,
             "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80",
             "euphoric|high-energy|triumphant|rock"),
            ("track_02", "Levitating", "Dua Lipa", "Future Nostalgia", "dance", 0.91, 0.83, 103.0, 0.70, 0.01, 92, 203000,
             "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=500&auto=format&fit=crop&q=80",
             "dance|disco|groove|positive"),
            ("track_03", "Good as Hell", "Lizzo", "Cuz I Love You", "pop", 0.89, 0.88, 96.0, 0.68, 0.18, 84, 159000,
             "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=80",
             "empowering|cheerful|confident|fun"),
            ("track_04", "Mr. Blue Sky", "Electric Light Orchestra", "Out of the Blue", "classic rock", 0.97, 0.89, 178.0, 0.38, 0.12, 86, 303000,
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80",
             "sunny|pure joy|optimistic|bright"),
            ("track_05", "Can't Stop the Feeling!", "Justin Timberlake", "Trolls OST", "pop", 0.91, 0.83, 113.0, 0.67, 0.01, 85, 236000,
             "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=500&auto=format&fit=crop&q=80",
             "joyous|danceable|infectious|upbeat"),
            ("track_06", "Levels", "Avicii", "Levels", "edm", 0.82, 0.94, 126.0, 0.58, 0.02, 87, 201000,
             "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500&auto=format&fit=crop&q=80",
             "festival|hype|anthem|euphoric"),
            ("track_07", "Happy", "Pharrell Williams", "G I R L", "pop", 0.96, 0.82, 160.0, 0.65, 0.22, 83, 233000,
             "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80",
             "cheerful|feel-good|catchy|sunshine"),
            ("track_08", "Midnight City", "M83", "Hurry Up, We're Dreaming", "synthwave", 0.72, 0.86, 105.0, 0.52, 0.01, 89, 244000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "nostalgic|neon|soaring|electrifying"),

            # High Valence, Low Energy (Cozy, Warm, Chill, Romantic, Peaceful)
            ("track_09", "Sunday Morning", "Maroon 5", "Songs About Jane", "pop rock", 0.88, 0.48, 88.0, 0.68, 0.32, 82, 242000,
             "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=500&auto=format&fit=crop&q=80",
             "warm|acoustic|coffee|peaceful"),
            ("track_10", "Sunset Lover", "Petit Biscuit", "Presence", "chillwave", 0.78, 0.44, 91.0, 0.78, 0.58, 84, 237000,
             "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=500&auto=format&fit=crop&q=80",
             "serene|golden hour|dreamy|breezy"),
            ("track_11", "Put Your Records On", "Corinne Bailey Rae", "Corinne Bailey Rae", "soul", 0.86, 0.52, 96.0, 0.65, 0.38, 81, 215000,
             "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=500&auto=format&fit=crop&q=80",
             "relaxing|summer|carefree|warm"),
            ("track_12", "Banana Pancakes", "Jack Johnson", "In Between Dreams", "acoustic", 0.82, 0.38, 110.0, 0.72, 0.81, 79, 192000,
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80",
             "lazy morning|sweet|mellow|rainy day"),
            ("track_13", "Here Comes the Sun", "The Beatles", "Abbey Road", "folk rock", 0.92, 0.45, 129.0, 0.56, 0.54, 91, 185000,
             "https://images.unsplash.com/photo-1509281373149-e957c6296406?w=500&auto=format&fit=crop&q=80",
             "hopeful|pastoral|timeless|comfort"),
            ("track_14", "Bloom", "The Paper Kites", "Woodland", "indie folk", 0.76, 0.32, 112.0, 0.59, 0.88, 77, 210000,
             "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=500&auto=format&fit=crop&q=80",
             "tender|acoustic|intimate|whispering"),
            ("track_15", "Beyond", "Leon Bridges", "Good Thing", "r-n-b", 0.79, 0.46, 85.0, 0.54, 0.42, 80, 240000,
             "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=500&auto=format&fit=crop&q=80",
             "romantic|velvet|soulful|gentle"),
            ("track_16", "Dreams", "Fleetwood Mac", "Rumours", "soft rock", 0.81, 0.49, 120.0, 0.83, 0.14, 93, 257000,
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80",
             "floating|smooth|timeless|groove"),

            # Low Valence, Low Energy (Sad, Melancholy, Ambient, Rainy, Heartbroken)
            ("track_17", "Holocene", "Bon Iver", "Bon Iver", "indie", 0.18, 0.32, 74.0, 0.38, 0.79, 82, 336000,
             "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=500&auto=format&fit=crop&q=80",
             "cathartic|vulnerable|wintry|spacious"),
            ("track_18", "Skinny Love", "Bon Iver", "For Emma, Forever Ago", "indie folk", 0.22, 0.28, 76.0, 0.45, 0.86, 83, 239000,
             "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=500&auto=format&fit=crop&q=80",
             "raw|heartbreak|acoustic|haunting"),
            ("track_19", "Weightless", "Marconi Union", "Weightless", "ambient", 0.12, 0.08, 60.0, 0.15, 0.95, 80, 485000,
             "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500&auto=format&fit=crop&q=80",
             "anti-anxiety|deep focus|floating|sleep"),
            ("track_20", "Gymnopédie No. 1", "Erik Satie", "Gymnopédies", "classical", 0.28, 0.09, 68.0, 0.32, 0.99, 81, 195000,
             "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?w=500&auto=format&fit=crop&q=80",
             "solitary|piano|introspective|poetic"),
            ("track_21", "Liability", "Lorde", "Melodrama", "pop", 0.33, 0.23, 76.0, 0.58, 0.89, 80, 172000,
             "https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=500&auto=format&fit=crop&q=80",
             "lonely|piano ballad|fragile|melodramatic"),
            ("track_22", "Motion Picture Soundtrack", "Radiohead", "Kid A", "art rock", 0.15, 0.21, 65.0, 0.24, 0.82, 75, 200000,
             "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500&auto=format&fit=crop&q=80",
             "ethereal|grief|transcendent|organ"),
            ("track_23", "Clair de Lune", "Claude Debussy", "Suite bergamasque", "classical", 0.29, 0.11, 62.0, 0.28, 0.98, 85, 304000,
             "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=500&auto=format&fit=crop&q=80",
             "lunar|impressionist|graceful|peaceful"),
            ("track_24", "The Night We Met", "Lord Huron", "Strange Trails", "indie folk", 0.25, 0.37, 87.0, 0.52, 0.65, 89, 208000,
             "https://images.unsplash.com/photo-1516589178581-6cd7833ae3b2?w=500&auto=format&fit=crop&q=80",
             "longing|echoing|haunted memory|ghostly"),

            # Low Valence, High Energy (Intense, Aggressive, Dark Electro, Anger, Workout, Tension)
            ("track_25", "Lose Yourself", "Eminem", "8 Mile OST", "hip-hop", 0.24, 0.93, 171.0, 0.69, 0.01, 91, 326000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "hyped|determined|intense|focus"),
            ("track_26", "Killing in the Name", "Rage Against The Machine", "RATM", "metal", 0.21, 0.96, 88.0, 0.44, 0.01, 85, 313000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "rage|raw power|rebellion|heavy"),
            ("track_27", "Starboy", "The Weeknd, Daft Punk", "Starboy", "synth-pop", 0.48, 0.82, 186.0, 0.68, 0.14, 94, 230000,
             "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500&auto=format&fit=crop&q=80",
             "dark pop|night drive|swagger|pulsing"),
            ("track_28", "Breathe", "The Prodigy", "The Fat of the Land", "breakbeat", 0.32, 0.98, 130.0, 0.64, 0.02, 78, 335000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "industrial|aggressive|adrenaline|club"),
            ("track_29", "Resonance", "HOME", "Odyssey", "synthwave", 0.42, 0.78, 170.0, 0.52, 0.05, 87, 212000,
             "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80",
             "cyberpunk|retro|neon highway|driving"),
            ("track_30", "Chop Suey!", "System of a Down", "Toxicity", "alternative metal", 0.31, 0.97, 127.0, 0.42, 0.01, 88, 210000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "explosive|frenetic|cathartic scream|rock"),
            ("track_31", "Blinding Lights", "The Weeknd", "After Hours", "synth-pop", 0.34, 0.73, 171.0, 0.51, 0.00, 96, 200000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "80s revival|driving fast|electric|nocturnal"),
            ("track_32", "In the End", "Linkin Park", "Hybrid Theory", "numetal", 0.39, 0.86, 105.0, 0.55, 0.01, 89, 216000,
             "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
             "angst|nostalgia|anthemic|piano rock")
        ]

        df_tracks = pd.DataFrame(tracks, columns=[
            "track_id", "track_name", "artists", "album_name", "track_genre",
            "valence", "energy", "tempo", "danceability", "acousticness",
            "popularity", "duration_ms", "cover_art", "mood_tags"
        ])
        df_tracks.to_csv(spotify_path, index=False)

    print("Datasets initialized successfully.")

def load_data():
    """Loads movies, ratings, and spotify tracks dataframes."""
    generate_default_datasets()
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    spotify_path = os.path.join(DATA_DIR, "spotify_tracks.csv")
    
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    df_tracks = pd.read_csv(spotify_path)
    return df_movies, df_ratings, df_tracks

if __name__ == "__main__":
    generate_default_datasets()
