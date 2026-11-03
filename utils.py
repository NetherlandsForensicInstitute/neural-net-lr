import numpy as np
from psycopg2 import sql

from DL_modules.database import DBConnection, fetch_all_results_from


def retrieve_data(db_connection: DBConnection, elements_to_retrieve: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve all ablations that are tagged as `6fold_measurements_lr_system`. Return only elements defined in
    `DEFAULT_LR_ELEMENTS`.

    As X, return a numpy array with ablations as rows and the measured elements as columns. As y, return a numpy
    1-D array with integers indicating the sample that the ablations belong to (so rows in X with the same integer
    at their index in y originate from the same sample).
    """
    query = sql.SQL("""
        select sample_id, measured_values, measured_elements
        from sample s
            join ablation a on s.id=a.sample_id and tag='6fold_measurements_lr_system'
            join measurement m on m.id=s.measurement_id;
    """)

    samples = fetch_all_results_from(db_connection, query)
    elements = samples[0]['measured_elements'].split(',')
    samples_values = [{e: v for v, e in zip(sample['measured_values'], elements)} for sample in samples]
    # make sure elements are in the correct order
    X = np.array([[sample[required_element] for required_element in elements_to_retrieve] for sample in samples_values])
    y = np.array([sample['sample_id'] for sample in samples])

    return X, y
