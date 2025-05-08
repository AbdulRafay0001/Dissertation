import math
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the database (for reference, though not used directly in the similarity calculation).
restaurant_data = {
    "The Green Spoon": "A modern eatery offering organic, farm-to-table dishes in a relaxed, eco-friendly environment.",
    "Urban Bites": "A trendy restaurant serving fusion cuisine that blends international flavors with local ingredients.",
    "Classic Diner": "An old-school diner known for hearty comfort food, a nostalgic atmosphere, and generous portions.",
    "Sea Breeze": "An elegant seafood restaurant with coastal decor and freshly caught dishes served in a serene setting.",
    "Spice Route": "An exotic spot featuring traditional Indian cuisine with a modern twist and vibrant, aromatic spices.",
    "Bella Italia": "A family-friendly Italian trattoria offering homemade pasta, wood-fired pizzas, and rustic charm.",
    "Tokyo House": "A cozy Japanese eatery specializing in sushi, sashimi, and comforting ramen bowls.",
    "The Smoke Pit": "A laid-back barbecue joint renowned for its slow-smoked meats, tangy sauces, and Southern sides.",
    "The Vegan Garden": "A plant-based cafe serving creative, locally sourced dishes in a casual, health-focused setting.",
    "Harvest & Hearth": "A farm-to-table bistro that emphasizes seasonal produce, artisanal breads, and a warm ambiance.",
    "Aurora Bistro": "A stylish bistro offering contemporary dishes with panoramic city views.",
    "Crescent Cafe": "A cozy cafe known for artisanal coffees, freshly baked pastries, and a welcoming atmosphere.",
    "Golden Spoon": "An elegant restaurant blending classic recipes with modern culinary techniques.",
    "The Rustic Fork": "A charming eatery featuring farm-sourced ingredients and hearty, homemade comfort food.",
    "Lakeside Grill": "A scenic grill by the lake serving fresh seafood and grilled specialties in a relaxed setting.",
    "Mountain Peak Diner": "A diner offering hearty meals perfect for refueling after outdoor adventures.",
    "City Lights Steakhouse": "A sophisticated steakhouse offering premium cuts and an energetic urban ambiance.",
    "Sunset Grill": "A casual grill with outdoor seating, ideal for enjoying the sunset and a laid-back meal.",
    "Moonlight Eatery": "An intimate eatery known for creative dishes and an enchanting nighttime setting.",
    "Harbor House": "A waterfront restaurant specializing in seafood and maritime-inspired cuisine.",
    "Valley Vista": "A family-friendly restaurant offering diverse dishes with breathtaking valley views.",
    "River's Edge": "A modern eatery by the river, known for its fresh and innovative menu.",
    "Urban Palette": "A chic restaurant celebrating diverse flavors with an artistic presentation.",
    "Global Grille": "A vibrant spot offering a fusion of international cuisines in a lively setting.",
    "Garden Feast": "A rustic restaurant set in a lush garden serving seasonal, locally sourced fare.",
    "The Daily Dish": "A casual dining spot delivering fresh, daily specials with creative twists.",
    "Nomad Nook": "An eclectic eatery offering a global menu in a warm and inviting space.",
    "Fiesta Feast": "A lively restaurant known for its festive ambiance and vibrant Latin flavors.",
    "Zen Garden": "A serene dining space blending Asian cuisine with calming, nature-inspired decor.",
    "Epicurean Delights": "A gourmet restaurant providing a feast for both the palate and the eyes.",
    "The Tasting Room": "An intimate venue for sampling a curated menu of innovative small plates.",
    "Saffron Lounge": "A stylish lounge offering Mediterranean-inspired dishes accented with aromatic spices.",
    "Crimson Kitchen": "A contemporary kitchen serving bold, flavorful dishes in a modern setting.",
    "Blue Ocean": "A refined seafood restaurant focusing on fresh, ocean-sourced ingredients.",
    "Emerald Eatery": "A vibrant restaurant known for its sustainable approach and creative, green-inspired menu.",
    "Pearl of the Sea": "A luxury seafood establishment delivering exquisite, ocean-fresh dishes.",
    "Modern Morsels": "A trendy spot offering bite-sized culinary innovations in a relaxed environment.",
    "Fusion Flame": "A dynamic restaurant blending Eastern and Western flavors with a fiery twist.",
    "Rustic Roots": "An inviting eatery rooted in traditional recipes and hearty, rustic flavors.",
    "Coastal Cravings": "A beachside restaurant serving fresh seafood and coastal favorites in a casual vibe.",
    "Palate Pleaser": "A diverse restaurant offering a wide range of dishes to satisfy every taste.",
    "Sunrise Diner": "An early-morning diner famous for its hearty breakfasts and warm, welcoming atmosphere.",
    "Twilight Tavern": "A cozy tavern offering hearty pub fare in a relaxed, twilight-inspired setting.",
    "Maple & Main": "A classic restaurant focusing on comfort food with a touch of elegant presentation.",
    "The Local Table": "A community-centric restaurant that celebrates local produce and regional flavors.",
    "Cityscape Cafe": "A modern cafe offering a vibrant urban dining experience with creative dishes.",
    "The Wanderlust": "An adventurous restaurant taking diners on a culinary journey around the world.",
    "Savory Junction": "A casual spot known for its delicious, savory dishes and friendly service.",
    "Taste of Tradition": "A restaurant that honors classic recipes while adding modern twists.",
    "Flavors & Fires": "A lively eatery featuring grilled specialties and bold, smoky flavors.",
    "The Spice Box": "A quaint restaurant focused on aromatic spices and flavor-rich dishes.",
    "Garden Gate": "A charming eatery set in a garden, serving fresh, organic meals.",
    "Wholesome Kitchen": "A health-focused cafe offering nutritious, delicious meals in a relaxed setting.",
    "Bistro Boulevard": "A chic bistro blending French culinary traditions with modern innovation.",
    "Culinary Canvas": "A creative restaurant where every dish is a work of art.",
    "Serenity Bites": "A peaceful eatery known for its light, refreshing dishes and calming atmosphere.",
    "The Urban Fork": "A trendy spot offering contemporary dishes in a sleek urban environment.",
    "Metropolitan Meals": "A cosmopolitan restaurant serving innovative cuisine in a stylish setting.",
    "Culinary Cove": "A hidden gem offering gourmet dishes in a relaxed, coastal atmosphere.",
    "Epic Eats": "A vibrant eatery delivering bold flavors and inventive dishes.",
    "Fusion Fare": "A modern restaurant merging international cuisines into unique fusion dishes.",
    "Sizzle & Savor": "A dynamic grill offering sizzling specialties paired with savory sides.",
    "Bountiful Bites": "A restaurant celebrated for its generous portions and diverse, flavorful menu.",
    "Infinite Flavors": "A culinary destination with an endless array of innovative dishes.",
    "The Gourmet Spot": "A refined restaurant dedicated to gourmet dishes and exquisite presentation.",
    "Crave Kitchen": "A lively spot serving crave-worthy dishes with a contemporary twist.",
    "Sunset Terrace": "An open-air restaurant perfect for dining under the evening sky.",
    "Harmony House": "A welcoming restaurant creating a balance between delicious flavors and a soothing ambiance.",
    "The Dining Den": "A cozy spot offering comforting fare in an intimate, den-like setting.",
    "Delish Deli": "A deli known for its mouthwatering sandwiches, salads, and fresh daily specials.",
    "Charming Chow": "A small restaurant with a warm ambiance and homestyle cooking.",
    "The Flavor Factory": "A creative kitchen crafting innovative dishes that delight the palate.",
    "Bold Bites": "An edgy restaurant offering bold, unconventional flavors in every dish.",
    "Aroma Avenue": "An inviting eatery known for its aromatic dishes and vibrant setting.",
    "Taste Trail": "A culinary journey in a restaurant offering diverse and rich flavors.",
    "The Culinary Corner": "A neighborhood favorite serving quality dishes with a personal touch.",
    "Hearty Haven": "A cozy spot renowned for its hearty meals and comforting environment.",
    "Divine Dine": "A high-end restaurant delivering divine flavors in an elegant atmosphere.",
    "Fresco Feasts": "A vibrant eatery where every meal feels like a fresh celebration.",
    "Culinary Crossroads": "A restaurant that brings together diverse culinary traditions in one place.",
    "The Rustic Plate": "A down-to-earth eatery emphasizing homemade, rustic dishes.",
    "Gastronomy Grove": "A modern restaurant that celebrates culinary art in a lush, green setting.",
    "The Palate Parlor": "A sophisticated restaurant designed to delight every sense.",
    "Savor Station": "A casual eatery known for its delicious, savory options and welcoming service.",
    "The Food Foundry": "An innovative restaurant forging new culinary experiences.",
    "Nourish Nook": "A healthy cafe focusing on wholesome, nourishing meals.",
    "Eclipse Eatery": "A trendy spot offering innovative dishes in a sleek, modern ambiance.",
    "The Savory Spot": "A well-loved restaurant delivering rich, savory flavors in every bite.",
    "Melody Meals": "A restaurant where every dish is prepared with rhythm and harmony.",
    "The Urban Diner": "A contemporary diner serving modern twists on classic comfort food.",
    "Whispering Willow": "A serene restaurant set amidst nature, offering dishes inspired by the outdoors.",
    "Opulent Oasis": "A luxurious restaurant providing opulent dining experiences and premium flavors.",
    "Modern Munchies": "A casual eatery focusing on creative, modern snacks and small plates.",
    "The Gourmet Garden": "A vibrant restaurant where fresh, garden-sourced ingredients shine.",
    "Culinary Quest": "A restaurant inviting you on a quest of diverse and exquisite tastes.",
    "The Harvest House": "A rustic restaurant emphasizing seasonal harvests and home-cooked meals.",
    "Rustic Retreat": "An inviting eatery combining rustic charm with contemporary comfort.",
    "Celestial Cuisine": "A high-end restaurant offering heavenly dishes in an ethereal setting.",
    "Tasteful Trails": "A restaurant that guides your taste buds along flavorful culinary paths.",
    "Vivid Vittles": "A lively eatery known for its colorful and inventive dishes.",
    "The Spice Symphony": "A dynamic restaurant where spices create a symphony of flavor.",
    "Gastronomic Gateway": "A trendy restaurant serving as a gateway to global flavors.",
    "The Fusion Forum": "A contemporary eatery merging diverse culinary traditions into one.",
    "Epicurean Edge": "A cutting-edge restaurant that challenges traditional culinary boundaries.",
    "Feast & Fable": "A whimsical restaurant where every meal tells a captivating story.",
    "The Flavorsmith": "A creative kitchen crafting unique and memorable dishes.",
    "Culinary Convergence": "A restaurant where diverse culinary styles converge into excellence.",
    "Savor Society": "A sophisticated eatery that celebrates the art of savoring food."
}


# Step 2: Define the attribute vectors for each perfume.
attribute_vectors = {
"The Green Spoon": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Urban Bites": {"diverse": 5, "flavorful": 5, "innovative": 4},
"Classic Diner": {"diverse": 2, "flavorful": 4, "innovative": 2},
"Sea Breeze": {"diverse": 3, "flavorful": 5, "innovative": 3},
"Spice Route": {"diverse": 4, "flavorful": 5, "innovative": 4},
"Bella Italia": {"diverse": 3, "flavorful": 4, "innovative": 2},
"Tokyo House": {"diverse": 3, "flavorful": 4, "innovative": 3},
"The Smoke Pit": {"diverse": 2, "flavorful": 5, "innovative": 3},
"The Vegan Garden": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Harvest & Hearth": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Aurora Bistro": {"diverse": 3, "flavorful": 4, "innovative": 4},
"Crescent Cafe": {"diverse": 2, "flavorful": 3, "innovative": 2},
"Golden Spoon": {"diverse": 3, "flavorful": 4, "innovative": 4},
"The Rustic Fork": {"diverse": 3, "flavorful": 4, "innovative": 2},
"Lakeside Grill": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Mountain Peak Diner": {"diverse": 2, "flavorful": 3, "innovative": 2},
"City Lights Steakhouse": {"diverse": 2, "flavorful": 4, "innovative": 3},
"Sunset Grill": {"diverse": 2, "flavorful": 3, "innovative": 2},
"Moonlight Eatery": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Harbor House": {"diverse": 3, "flavorful": 5, "innovative": 3},
"Valley Vista": {"diverse": 4, "flavorful": 4, "innovative": 3},
"River's Edge": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Urban Palette": {"diverse": 5, "flavorful": 4, "innovative": 5},
"Global Grille": {"diverse": 5, "flavorful": 4, "innovative": 4},
"Garden Feast": {"diverse": 3, "flavorful": 4, "innovative": 3},
"The Daily Dish": {"diverse": 3, "flavorful": 4, "innovative": 4},
"Nomad Nook": {"diverse": 5, "flavorful": 4, "innovative": 4},
"Fiesta Feast": {"diverse": 4, "flavorful": 5, "innovative": 3},
"Zen Garden": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Epicurean Delights": {"diverse": 3, "flavorful": 5, "innovative": 5},
"The Tasting Room": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Saffron Lounge": {"diverse": 4, "flavorful": 5, "innovative": 4},
"Crimson Kitchen": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Blue Ocean": {"diverse": 3, "flavorful": 5, "innovative": 3},
"Emerald Eatery": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Pearl of the Sea": {"diverse": 3, "flavorful": 5, "innovative": 3},
"Modern Morsels": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Fusion Flame": {"diverse": 5, "flavorful": 4, "innovative": 5},
"Rustic Roots": {"diverse": 3, "flavorful": 4, "innovative": 2},
"Coastal Cravings": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Palate Pleaser": {"diverse": 5, "flavorful": 5, "innovative": 4},
"Sunrise Diner": {"diverse": 2, "flavorful": 3, "innovative": 2},
"Twilight Tavern": {"diverse": 2, "flavorful": 4, "innovative": 2},
"Maple & Main": {"diverse": 2, "flavorful": 4, "innovative": 2},
"The Local Table": {"diverse": 4, "flavorful": 4, "innovative": 3},
"Cityscape Cafe": {"diverse": 4, "flavorful": 4, "innovative": 4},
"The Wanderlust": {"diverse": 5, "flavorful": 5, "innovative": 5},
"Savory Junction": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Taste of Tradition": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Flavors & Fires": {"diverse": 3, "flavorful": 5, "innovative": 4},
"The Spice Box": {"diverse": 3, "flavorful": 5, "innovative": 3},
"Garden Gate": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Wholesome Kitchen": {"diverse": 3, "flavorful": 4, "innovative": 4},
"Bistro Boulevard": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Culinary Canvas": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Serenity Bites": {"diverse": 2, "flavorful": 3, "innovative": 2},
"The Urban Fork": {"diverse": 4, "flavorful": 4, "innovative": 4},
"Metropolitan Meals": {"diverse": 4, "flavorful": 4, "innovative": 5},
"Culinary Cove": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Epic Eats": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Fusion Fare": {"diverse": 5, "flavorful": 4, "innovative": 5},
"Sizzle & Savor": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Bountiful Bites": {"diverse": 4, "flavorful": 5, "innovative": 3},
"Infinite Flavors": {"diverse": 5, "flavorful": 5, "innovative": 5},
"The Gourmet Spot": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Crave Kitchen": {"diverse": 4, "flavorful": 4, "innovative": 4},
"Sunset Terrace": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Harmony House": {"diverse": 3, "flavorful": 4, "innovative": 3},
"The Dining Den": {"diverse": 2, "flavorful": 3, "innovative": 2},
"Delish Deli": {"diverse": 2, "flavorful": 4, "innovative": 2},
"Charming Chow": {"diverse": 2, "flavorful": 3, "innovative": 2},
"The Flavor Factory": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Bold Bites": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Aroma Avenue": {"diverse": 4, "flavorful": 4, "innovative": 3},
"Taste Trail": {"diverse": 5, "flavorful": 5, "innovative": 4},
"The Culinary Corner": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Hearty Haven": {"diverse": 2, "flavorful": 4, "innovative": 2},
"Divine Dine": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Fresco Feasts": {"diverse": 4, "flavorful": 5, "innovative": 4},
"Culinary Crossroads": {"diverse": 5, "flavorful": 5, "innovative": 5},
"The Rustic Plate": {"diverse": 2, "flavorful": 3, "innovative": 2},
"Gastronomy Grove": {"diverse": 4, "flavorful": 5, "innovative": 4},
"The Palate Parlor": {"diverse": 3, "flavorful": 5, "innovative": 5},
"Savor Station": {"diverse": 3, "flavorful": 4, "innovative": 3},
"The Food Foundry": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Nourish Nook": {"diverse": 3, "flavorful": 4, "innovative": 4},
"Eclipse Eatery": {"diverse": 4, "flavorful": 4, "innovative": 5},
"The Savory Spot": {"diverse": 2, "flavorful": 4, "innovative": 3},
"Melody Meals": {"diverse": 3, "flavorful": 4, "innovative": 3},
"The Urban Diner": {"diverse": 2, "flavorful": 4, "innovative": 3},
"Whispering Willow": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Opulent Oasis": {"diverse": 3, "flavorful": 5, "innovative": 4},
"Modern Munchies": {"diverse": 3, "flavorful": 4, "innovative": 4},
"The Gourmet Garden": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Culinary Quest": {"diverse": 5, "flavorful": 5, "innovative": 5},
"The Harvest House": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Rustic Retreat": {"diverse": 3, "flavorful": 4, "innovative": 3},
"Celestial Cuisine": {"diverse": 3, "flavorful": 5, "innovative": 5},
"Tasteful Trails": {"diverse": 4, "flavorful": 4, "innovative": 4},
"Vivid Vittles": {"diverse": 4, "flavorful": 5, "innovative": 5},
"The Spice Symphony": {"diverse": 4, "flavorful": 5, "innovative": 4},
"Gastronomic Gateway": {"diverse": 5, "flavorful": 5, "innovative": 5},
"The Fusion Forum": {"diverse": 5, "flavorful": 4, "innovative": 5},
"Epicurean Edge": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Feast & Fable": {"diverse": 4, "flavorful": 4, "innovative": 5},
"The Flavorsmith": {"diverse": 4, "flavorful": 5, "innovative": 5},
"Culinary Convergence": {"diverse": 5, "flavorful": 5, "innovative": 5},
"Savor Society": {"diverse": 4, "flavorful": 4, "innovative": 4}
}


# Step 3: Define the user preference vector in the order [timeless, floral, luxurious].
user_preference_vector = [2, 5 ,2]

# Helper function to convert a perfume's attribute dictionary into a list 
# matching the same attribute order as the user preference vector.
def attribute_dict_to_list(attr_dict):
    return [
        attr_dict["diverse"],
        attr_dict["flavorful"],
        attr_dict["innovative"]
    ]

# Function to calculate cosine similarity between two vectors.
def cosine_similarity(vecA, vecB):
    dot_product = sum(a * b for a, b in zip(vecA, vecB))
    normA = math.sqrt(sum(a * a for a in vecA))
    normB = math.sqrt(sum(b * b for b in vecB))
    return dot_product / (normA * normB)

# Step 4: Calculate the similarity for each perfume and track the best match.
similarity_scores = []  # New list to store all similarity scores
highest_similarity = -1
best_product = None


for perfume, attrs in attribute_vectors.items():
    product_vector = attribute_dict_to_list(attrs)
    similarity = cosine_similarity(product_vector, user_preference_vector)
    similarity_scores.append(similarity)
    print(f"{perfume}: similarity = {similarity:.2f}")
    if similarity > highest_similarity:
        highest_similarity = similarity
        best_product = perfume

# Step 5: Print the recommendation.
print(f"\nRecommended product: {best_product} (similarity = {highest_similarity:.2f})")

# Calculate mean and standard deviation of similarity scores
mean_similarity = np.mean(similarity_scores)
std_similarity = np.std(similarity_scores)

# Create a histogram of cosine similarity scores
plt.figure(figsize=(10, 5))
plt.hist(similarity_scores, bins=10, edgecolor='black')
plt.title("Histogram of Cosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")

# Draw lines for mean and ±1 standard deviation
plt.axvline(mean_similarity, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_similarity:.2f}')
plt.axvline(mean_similarity + std_similarity, color='green', linestyle='dashed', linewidth=1, label=f'+1 Std: {mean_similarity + std_similarity:.2f}')
plt.axvline(mean_similarity - std_similarity, color='green', linestyle='dashed', linewidth=1, label=f'-1 Std: {mean_similarity - std_similarity:.2f}')
plt.legend()
plt.show()


# Create a boxplot of the cosine similarity scores
plt.figure(figsize=(5, 5))
plt.boxplot(similarity_scores, patch_artist=True)
plt.title("Boxplot of Cosine Similarities")
plt.ylabel("Cosine Similarity")
plt.show()