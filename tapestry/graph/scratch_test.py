from tapestry.graph import docs


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
