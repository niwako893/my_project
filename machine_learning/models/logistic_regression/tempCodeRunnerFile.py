sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from plot_utils import plot_regression_line
from plot_utils import plot_classification_report_bar
from plot_utils import plot_confusion_matrix_heatmap



df = pd.read_csv("data/survey lung cancer.csv")