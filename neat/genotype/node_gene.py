
class NodeGene:
    def __init__(self, node_id, node_type, activation=None):
        self.id = node_id
        self.type = node_type
        self.unit = None
        self.activation = activation

    def __str__(self):
        return str(self.id) + '-' + self.type + '-' + str(self.activation.__name)
