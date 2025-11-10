import json
from cgshop2026_pyutils.schemas import CGSHOP2026Instance


class CGSHOP2026InstanceParser:
    """
    A parser for CGSHOP 2026 Instance JSON files.
    
    Reads an instance file, creates a CGSHOP2026Instance object,
    and provides methods to access triangulations and related data.
    """

    def __init__(self, json_path: str):
        """
        Initialize the parser with the path to the JSON file.
        """
        self.json_path = json_path
        self.data = None
        self.instance = None

    def load(self):
        """
        Load the JSON data and create a CGSHOP2026Instance object.
        """
        with open(self.json_path, "r") as file:
            self.data = json.load(file)

        self.instance = CGSHOP2026Instance(
            instance_uid=self.data["instance_uid"],
            points_x=self.data["points_x"],
            points_y=self.data["points_y"],
            triangulations=self.data["triangulations"]
        )

    def get_triangulations(self):
        """
        Return the list of triangulations.
        """
        if not self.instance:
            raise ValueError("Instance not loaded. Call load() first.")
        return self.instance.triangulations

    def print_triangulations(self):
        """
        Print all triangulations and their edges.
        """
        triangulations = self.get_triangulations()
        print(f"Instance UID: {self.instance.instance_uid}")
        print(f"Number of triangulations: {len(triangulations)}\n")

        for i, tri in enumerate(triangulations):
            print(f"Triangulation {i}:")
            for edge in tri:
                print("  ", tuple(edge))
            print()


# Example usage
if __name__ == "__main__":
    parser = CGSHOP2026InstanceParser("./example_instances_cgshop2026/examples/example_ps_20_nt2_pfd5_random.json")
    parser.load()
    parser.print_triangulations()
