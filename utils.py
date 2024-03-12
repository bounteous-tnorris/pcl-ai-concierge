from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform_v1
import json
import os 


def text_embedding(input) -> list:
    """Text embedding with a Large Language Model."""

    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    embeddings = model.get_embeddings(input)
    for embedding in embeddings:
        vector = embedding.values
        # print(f"Length of Embedding Vector: {len(vector)}")
    return vector 


def base_search(api_endpoint, index_endpoint, deployed_index, emb_prompt):
    # Configure Vector Search client
    client_options = {
      "api_endpoint": api_endpoint
    }
    vector_search_client = aiplatform_v1.MatchServiceClient(
      client_options=client_options,
    )

    # Build FindNeighborsRequest object
    datapoint = aiplatform_v1.IndexDatapoint(
      feature_vector=emb_prompt
    )
    query = aiplatform_v1.FindNeighborsRequest.Query(
      datapoint=datapoint,
      # The number of nearest neighbors to be retrieved
      neighbor_count=10
    )
    request = aiplatform_v1.FindNeighborsRequest(
      index_endpoint=index_endpoint,
      deployed_index_id=deployed_index,
      # Request can have multiple queries
      queries=[query],
      return_full_datapoint=False,
    )

    # Execute the request
    return vector_search_client.find_neighbors(request)


def get_neighbors(response, full_json, text_field_name):
    neighbors = []

    for r in response.nearest_neighbors:

        for n in r.neighbors:
            # print(matching_data[int(n.datapoint.datapoint_id)]['text'])
            datapoint_id =  int(n.datapoint.datapoint_id)  
            neighbors.append(full_json[datapoint_id][text_field_name])

        neighbors = list(set(neighbors))

    # for i in neighbors:
    #     print(i)

    return neighbors  # '\n\n'.join(neighbors)


def search_summaries(emb_prompt):
    results = base_search(SUMMARY_API_ENDPOINT, SUMMARY_INDEX_ENDPOINT, SUMMARY_DEPLOYED_INDEX_ID, emb_prompt)
    # look up the indexes from results using the 'all_text' field in the embedding_outputs/summaries.json file
    return get_neighbors(results, summaries_json, 'all_text')


def combine_results(summary_results, review_results):
    return {
        'summary': summary_results,
        'review': review_results
    }


def do_search(prompt):
    # init()  # read in the json files and get the gecko model for embedding our prompt
    
    # convert prompt to embedding HERE
    emb_prompt = text_embedding([prompt])
    
    summary_results = search_summaries(emb_prompt)
    review_results = search_reviews(emb_prompt)
    
    # then put them together here and return the result
    return combine_results(summary_results, review_results)

  

d = do_search('Jamacian bike ride')
print(d)