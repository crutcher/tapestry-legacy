from tapestry.graph import docs
import uuid

import networkx as nx

def test_scratch():
    g = docs.TapestryGraphDoc()

    x = g.add_node(
        docs.TapestryNodeDoc(
            type="blockop",
            fields={
                "operation": "matmul",
                "index_space": [3, 4],
            },
        )
    )

    print(g.to_json_str(indent=2))


    a = uuid.uuid4()
    brand_x = docs.TapestryTargetDoc.from_json_data({ '_target_': a })
    print(brand_x)

    g = nx.DiGraph(id=uuid.uuid4())
    a = uuid.uuid4()
    b = uuid.uuid4()
    g.add_node(a)
    g.add_node(b)
    g.add_edge(a, b)
    print(g)