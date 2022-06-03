import os.path as op
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from helper.helper import *

from toolbox_ui import FeatureExtraction, RDM
from toolbox_ui import create_json
from evaluation import Evaluation
from helper.helper_ui import *

"""Write down all relevant paths"""
PATH_COLLECTION = get_paths()
CURRENT_DIR = PATH_COLLECTION["CURRENT_DIR"]
BASE_DIR = PATH_COLLECTION["BASE_DIR"]
GUI_DIR = PATH_COLLECTION["GUI_DIR"]
PARENT_DIR = PATH_COLLECTION["PARENT_DIR"]
INPUTS_DIR = PATH_COLLECTION["INPUTS_DIR"]
FEATS_DIR = PATH_COLLECTION["FEATS_DIR"]
RDMS_DIR = PATH_COLLECTION["RDMS_DIR"]
STIMULI_DIR = PATH_COLLECTION["STIMULI_DIR"]
BRAIN_DIR = PATH_COLLECTION["BRAIN_DIR"]



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        """Load GUI"""
        gui_filepath = op.join(GUI_DIR, 'mainwindow.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)

        """Connect buttons"""
        self.btn_feats.clicked.connect(self.clicked_genfeats)
        self.btn_rdm.clicked.connect(self.clicked_rdm)
        self.btn_eval.clicked.connect(self.clicked_eval)
        self.btn_exit.clicked.connect(self.end)

    """Generating Features"""

    def clicked_genfeats(self):
        """ Generate Features GUI"""
        gui_filepath = op.join(GUI_DIR, 'genfeats.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)

        """Connect buttons to functions"""
        self.btn_start.clicked.connect(self.start_genfeats)
        self.btn_exit.clicked.connect(self.load_start)

        """Fill dataset combobox"""
        stimuli_path = op.join(INPUTS_DIR, 'stimuli_data')
        available_images = findfolders(stimuli_path)
        self.cb_dataset.addItems(available_images)

        """Make dependent combo boxes for the networks"""
        netsets = get_available_nets()
        for set, networks in netsets.items():
            self.cb_netset.addItem(set, networks)

        self.cb_netset.currentIndexChanged.connect(self.updateGenSetCombo)
        self.updateGenSetCombo(self.cb_netset.currentIndex())
        self.show()

    def updateGenSetCombo(self, index):
        """Dependant Combobox Updater

        Args:
            index (int): [index of item in combobox]
        """

        self.cb_net.clear()
        networks = self.cb_netset.itemData(index)

        if networks:
            self.cb_net.addItems(networks)

    def start_genfeats(self):
        """Grab content and start generation """
        dataset = str(self.cb_dataset.currentText())
        netset = str(self.cb_netset.currentText())
        network = str(self.cb_net.currentText())
        
        """exclude amount from set name"""
        network_set = str(netset.split(" ")[0])
        self.load_loadingscreen()
        extract = FeatureExtraction(network, dataset, network_set)
        extract.start_extraction()

        print("")
        print("####################")
        print("Done")
        print("####################")
        print("")
        self.load_start()

    """Creating RDMs"""

    def clicked_rdm(self):
        """RDMs GUI"""
        
        """Load GUI"""
        gui_filepath = op.join(GUI_DIR, 'rdms.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)

        """Connect buttons"""
        self.btn_start.clicked.connect(self.clicked_start_rdm)
        self.btn_exit.clicked.connect(self.load_start)

        """Prepare dictionary for dependent combobox"""
        dict_rdm_net = {}
        available_datasets = findfolders(FEATS_DIR)

        for dataset in available_datasets:
            feats_path = op.join(FEATS_DIR, dataset)
            available_networks = findfolders(feats_path)
            dict_rdm_net.update({dataset: available_networks})  # This collects all available networks per dataset

        """Make dependent combo boxes for the datasets"""
        for set, networks in dict_rdm_net.items():
            self.cb_dataset.addItem(set, networks)

        self.cb_dataset.currentIndexChanged.connect(self.updateRDMSetCombo)
        self.updateRDMSetCombo(self.cb_dataset.currentIndex())
        self.show()

    def updateRDMSetCombo(self, index):
        """Dependant Combobox Updater

        Args:
            index (int): [index of item in combobox]
        """

        self.cb_netset.clear()
        datasets = self.cb_dataset.itemData(index)

        if datasets:
            self.cb_netset.addItems(datasets)

    def clicked_start_rdm(self):
        """Grab content and start creating RDMs"""

        dataset = str(self.cb_dataset.currentText())
        network = str(self.cb_netset.currentText())
        
        save_path = op.join(RDMS_DIR, dataset, network)
        feats_data_path = op.join(FEATS_DIR, dataset, network)

        self.load_loadingscreen()
        
        rdm = RDM(save_path, feats_data_path)
        rdm.create_rdms()

        print("")
        print("####################")
        print("Done")
        print("####################")
        print("")
        self.load_start()

    "Evaluation"

    def clicked_eval(self):
        
        """Load GUI"""
        gui_filepath = op.join(GUI_DIR, 'evalu.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)
        self.btn_start.setEnabled(True)
        
        """Fill metric combobox"""
        available_metrics = get_available_metrics()
        self.cb_dataset_2.addItems(available_metrics)
        
        """Fill dataset combobox"""
        available_rdms = findfolders(RDMS_DIR)
        self.cb_dataset.addItems(available_rdms)

        """Buttons"""
        self.btn_start.clicked.connect(self.clicked_eval2)
        self.btn_exit.clicked.connect(self.load_start)

        self.show()

    """Evaluation 2"""

    def clicked_eval2(self):
        """Evaluation 2 GUI"""

        """Grab content from before"""
        self.this_metric = str(self.cb_dataset_2.currentText())
        self.this_dataset = str(self.cb_dataset.currentText())
      
        """Load GUI"""
        gui_filepath = op.join(GUI_DIR, 'eval2.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)

        # Dont enable button if no selcations have been made
        self.btn_start.setEnabled(False)
        
        # Find selectable networks
        self.available_networks_eval = findfolders(op.join(RDMS_DIR, self.this_dataset))
        self.this_selected_nets = []

        for network in self.available_networks_eval:
            self.lst_dataset.addItem(network)
        
        ''' Connect buttons'''
        self.btn_start.clicked.connect(self.clicked_eval3)
        self.btn_exit.clicked.connect(self.load_start)
        self.btn_add.clicked.connect(self.eval2_add)


    def eval2_add(self):
        """Add items to list"""

        self.btn_start.setEnabled(True)  # Only enable as soon as nets are added
        
        # add items to the dict_selected for use in the next function
        selected = [item.text() for item in self.lst_dataset.selectedItems()]
  
        # Add it to global list
        for element in selected:
            self.this_selected_nets.append(element)

        # Add selected networks to the field of chosen networks
        for network in selected:
            self.lst_selection.addItem(network)
                        
                        
        # Remove already selected data to the selection field
        self.lst_dataset.clear()
        for network in self.available_networks_eval:
            if network not in self.this_selected_nets:
                self.lst_dataset.addItem(network)
        

    """Evaluation 3"""

    def clicked_eval3(self):
        """Evaluation 3 GUI"""

        """Load GUI"""
        gui_filepath = op.join(GUI_DIR, 'eval3.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)

        # Dont enable button if no selcations have been made
        self.btn_start.setEnabled(False)

        # Find selectable networks
        self.available_ROIs = findfilesonly(op.join(BRAIN_DIR, self.this_dataset), type="npz")
        self.this_selected_ROIs = []

        for roi in self.available_ROIs:
            self.lst_dataset.addItem(roi)

        ''' Connect buttons'''
        self.btn_start.clicked.connect(self.clicked_start_eval)
        self.btn_exit.clicked.connect(self.load_start)
        self.btn_add.clicked.connect(self.eval3_add)

    def eval3_add(self):
        """Add items to list"""

        # Only enable as soon as nets are added
        self.btn_start.setEnabled(True)

        # add items to the dict_selected for use in the next function
        selected = [item.text() for item in self.lst_dataset.selectedItems()]

        # Add it to global list
        for element in selected:
            self.this_selected_ROIs.append(element)

        # Add selected networks to the field of chosen networks
        for roi in selected:
            self.lst_selection.addItem(roi)

        # Remove already selected data to the selection field
        self.lst_dataset.clear()
        for roi in self.available_ROIs:
            if roi not in self.this_selected_ROIs:
                self.lst_dataset.addItem(roi)


    def clicked_start_eval(self):
        """Create a dict that is like {Link to ROI: Links to Model RDMs}"""
        
        json_dir = create_json(self.this_selected_nets, self.this_dataset, self.this_selected_ROIs, self.this_metric)
        
        evaluator = Evaluation(json_dir)
        evaluator.show_results()

        self.load_start()

    """Functions for loading screen an main menu"""

    def load_loadingscreen(self):
        gui_filepath = op.join(GUI_DIR, 'loading.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)
        self.show()
        QApplication.processEvents()

    def load_start(self):
        os.chdir(BASE_DIR)
        gui_filepath = op.join(GUI_DIR, 'mainwindow.ui')
        loadUi(gui_filepath, self)
        self.setMinimumSize(700, 750)

        self.btn_feats.clicked.connect(self.clicked_genfeats)  # connect to function
        self.btn_rdm.clicked.connect(self.clicked_rdm)
        self.btn_eval.clicked.connect(self.clicked_eval)
        self.btn_exit.clicked.connect(self.end)

    def end(self):
        self.close()
        sys.exit()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setWindowTitle("net2brain")
    brain_icon_path = op.join(GUI_DIR, "Brain.png")
    widget.setWindowIcon(QIcon(brain_icon_path))

    widget.show()
    sys.exit(app.exec_())
