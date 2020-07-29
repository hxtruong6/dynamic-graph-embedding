class CheckpointConfig:
    def __init__(self, number_saved=10, folder_path=None, index=None):
        self.NumberSaved = number_saved
        self.FolderPath = folder_path
        self.Index = index
