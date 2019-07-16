import csv
import sys
import os
import json
sys.path.insert(0, os.path.abspath('..'))
from Serializable.SerializableToModel import SerializableToModelConverter

converter = SerializableToModelConverter()
fileName = sys.argv[1]
#synonymsCs = sys.argv[2]
data = converter.convertFromFile(fileName)
attributes = ["anime_id", "anime_index", "anime_english_title", "anime_image_url", "anime_type", "anime_aired", "anime_rating_value"]
result = []
#synonymsDict = {}
#with open(synonymsCsv, "rb") as videocsv:
#        reader = csv.DictReader(videocsv)
#        for row in reader:
#            anime_id = row["anime_id"]
#            if not anime_id.isdigit():
#                print("Skipping: {}".format(anime_id))
#                continue
#            anime_id = int(anime_id)
#            anime_video_url = row["anime_video_url"]
#            if "not available" in anime_video_url:
#                videoUrlDict[anime_id] = "none"
#            else:
#                videoUrlDict[anime_id] = anime_video_url
#result = {}
for document in data:
    dictionary = {attribute:getattr(document, attribute) for attribute in attributes}
#    anime_id = int(dictionary["anime_id"])
#    dictionary["anime_id"] = int(anime_id)
    #try:
    #    dictionary["anime_english_title"] = str(dictionary["anime_english_title"])
    #    if len(dictionary["anime_english_title"]):
    #        dictionary["anime_english_title"] = document.anime_title
    #except:
    #    pass
    if dictionary["anime_english_title"] == "":
        print("Replacing anime id {}".format(document.anime_id))
        dictionary["anime_english_title"] = document.anime_title
#    if anime_id in videoUrlDict:
#        dictionary["anime_video_url"] = videoUrlDict[anime_id]
#    else:
#        dictionary["anime_video_url"] = "none"
#    try:
#        dictionary["anime_ranked"] = int(dictionary["anime_ranked"][1:])
 #   except:
 #       pass
 #   try:
#        dictionary["anime_popularity"] = int(dictionary["anime_popularity"][1:])
#    except:
#        pass
#    dictionary["anime_review_overall_average"] = document.getReviewOverallAverage()
#    dictionary["anime_review_story_average"] = document.getReviewStoryAverage()
#    dictionary["anime_review_animation_average"] = document.getReviewAnimationAverage()
#    dictionary["anime_review_sound_average"] = document.getReviewSoundAverage()
#    dictionary["anime_review_character_average"] = document.getReviewCharacterAverage()
#    dictionary["anime_review_enjoyment_average"] = document.getReviewEnjoymentAverage()
#    dictionary["anime_tags"] = "|".join(document.getAllTags())
#    result[int(anime_id)] = dictionary
#    print(anime_id)
    result.append(dictionary)
print("Found {} anime".format(len(result)))
with open("../../app/static/data/anime_search.json", "w+") as jsonFile:
    #for resultDict in result:
    json.dump(result, jsonFile, indent=4)
    #    outfile.write('\n')
