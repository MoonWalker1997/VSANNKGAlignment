import json

package_files = []

package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_1_university_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_2_musicalwork_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_3_airport_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_4_building_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_5_athlete_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_6_politician_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_7_company_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_8_celestialbody_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_9_astronaut_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_10_comicscharacter_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_11_meanoftransportation_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_12_monument_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_13_food_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_14_writtenwork_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_15_sportsteam_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_16_city_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_17_artist_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_18_scientist_ground_truth.jsonl")
package_files.append("/Users/tory/Downloads/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_19_film_ground_truth.jsonl")

Es = []
Rs = []

for package_file in package_files:

    g_entities = set()
    g_relations = set()

    package_data = []

    with open(package_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tmp = [data["sent"], []]
            for each in data["triples"]:
                g_entities.add(each["sub"])
                g_relations.add(each["rel"])
                g_entities.add(each["obj"])
                tmp[1].append([each["sub"], each["rel"], each["obj"]])
            package_data.append(tmp)

    Es.append(len(g_entities))
    Rs.append(len(g_relations))

print(Es)
print(Rs)
