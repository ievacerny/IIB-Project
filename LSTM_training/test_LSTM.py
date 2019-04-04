"""Unit and functional tests for LSTM_train script."""
import numpy as np
from os.path import join as pjoin
import unittest

import LSTM_train


class TestLoadFiles(unittest.TestCase):
    """Test function loadFiles that loads gesture and label data from files."""

    def setUp(self):
        """Set up all the tests."""
        self.true_labels = np.array([
            0, 2, 2, 2, 0,
            1, 1, 1,
            3, 3, 3,
            0, 102, 102, 102, 0,
            101, 101, 101], dtype='int32')
        self.true_gestures = np.ones((19, 249))
        self.true_gestures = (
            self.true_gestures * self.true_labels[:, np.newaxis])
        self.true_labels = self.true_labels.tolist()

    def test_loadall_success(self):
        """Test if files load successfully."""
        # Given
        file_order = [2, 1, 3, 102, 101]
        data_folder = pjoin("..", "Database", "TestDatabase")
        # When
        gestures, labels = LSTM_train.loadFiles(
            file_order, data_folder=data_folder)
        # Then
        self.assertEqual(19, len(labels))
        self.assertTupleEqual((19, 249), gestures.shape)

        self.assertListEqual(self.true_labels, labels.tolist())
        np.testing.assert_allclose(self.true_gestures, gestures)

    def test_loadall_long(self):
        """Test for exception if labels list is longer than gestures list."""
        # Video 4 is same as 3, but with an error in labels
        # Given
        file_order = [2, 1, 4, 102, 101]
        data_folder = pjoin("..", "Database", "TestDatabase")
        # When & Then
        with self.assertRaisesRegex(
                Exception,
                "Label and gesture files don't match in length"):
            gestures, labels = LSTM_train.loadFiles(
                file_order, data_folder=data_folder)

    def test_loadall_short(self):
        """Test for exception if labels list is shorter than gestures list."""
        # Video 5 is same as 3, but with an error in labels
        file_order = [2, 1, 5, 102, 101]
        data_folder = pjoin("..", "Database", "TestDatabase")
        # Error has to do with trying to fit in bigger array into smaller
        # (shape broadcasting)
        with self.assertRaises(ValueError):
            gestures, labels = LSTM_train.loadFiles(
                file_order, data_folder=data_folder)


class TestLoadData(unittest.TestCase):
    """Test load Data function that splits training and testing.

    Dependency: loadFiles
    """

    def setUp(self):
        """Set up all the tests."""
        self.seed = 3  # For this seed file_numbers = [2, 1, 3, 101, 102]
        self.number_of_vids = (3+1, 102+1)
        self.data_folder = pjoin("..", "Database", "TestDatabase")
        self.labels = [
            0, 2, 2, 2, 0,
            1, 1, 1,
            3, 3, 3,
            101, 101, 101,
            0, 102, 102, 102, 0]

    def test_load_data(self):
        """Test successful loading and splitting."""
        params = [  # testing_prop, exp_training_len, exp_testing_len
            (0.2, 14, 5),
            (0.5, 11, 8),
            (0.55, 8, 11),
        ]
        for testing_prop, exp_training_len, exp_testing_len in params:
            with self.subTest(testing_prop=testing_prop):
                # Given
                np.random.seed(self.seed)
                # When
                training, testing = LSTM_train.loadData(
                    testing_prop, self.number_of_vids, self.data_folder)
                # Then
                self.assertEqual(2, len(training))
                self.assertEqual(exp_training_len, len(training[0]))
                self.assertListEqual(
                    self.labels[exp_testing_len:], training[1].tolist())

                self.assertEqual(2, len(testing))
                self.assertEqual(exp_testing_len, len(testing[0]))
                self.assertListEqual(
                    self.labels[:exp_testing_len], testing[1].tolist())

    def test_load_data_exception(self):
        """Test when there are no testing or no training videos."""
        params = [  # testing_prop, msg
            (0, "No videos for testing."),
            (0.1, "No videos for testing."),
            (0.95, "No videos for training."),
            (1, "No videos for training."),
        ]
        for testing_prop, msg in params:
            with self.subTest(testing_prop=testing_prop, msg=msg):
                # Given
                np.random.seed(self.seed)
                # When & Then
                with self.assertRaisesRegex(Exception, msg):
                    training, testing = LSTM_train.loadData(
                        testing_prop, self.number_of_vids, self.data_folder)


class TestGetWindowStart(unittest.TestCase):
    """Test getWindowStart generator that randomises the input data.

    Dependency: loadData
    """

    def setUp(self):
        """Set up all the tests."""
        self.seed = 3
        number_of_vids = (3+1, 102+1)
        data_folder = pjoin("..", "Database", "TestDatabase")
        testing_prop = 0.2
        np.random.seed(self.seed)
        training, testing = LSTM_train.loadData(
            testing_prop, number_of_vids, data_folder)
        LSTM_train.training_data = training

    def test_get_window_start_success(self):
        """Test if window_start positions are generated correctly."""
        params = [  # n_frames, frame_step, expected_values
            (1, 1, [7, 4, 1, 2, 13, 6, 5, 0, 11, 12, 3, 9, 8, 10]),
            (3, 1, [5, 4, 1, 2, 11, 6, 7, 0, 3, 9, 8, 10]),
            (3, 2, [5, 4, 1, 2, 9, 6, 7, 0, 3, 8]),
            (2, 5, [6, 2, 7, 4, 5, 0, 3, 1, 8]),
        ]
        for n_frames, frame_step, expected_values in params:
            with self.subTest(n_frames=n_frames, frame_step=frame_step):
                # Given
                config = LSTM_train.Config(
                    n_frames=n_frames, frame_step=frame_step)
                np.random.seed(self.seed)
                # When
                train_gen = LSTM_train.getWindowStart(config)
                values = []
                for i in range(len(expected_values)):
                    values.append(next(train_gen))
                # Then
                self.assertListEqual(expected_values, values)

    def test_get_window_start_not_enough_frames(self):
        """Test for cases that should cause exceptions."""
        params = [  # n_frames, frame_step, description
            (20, 1, "Note enough frames"),
            (2, 20, "Frame step size too big"),
        ]
        for n_frames, frame_step, description in params:
            with self.subTest(msg=description):
                # Given
                config = LSTM_train.Config(
                    n_frames=n_frames, frame_step=frame_step)
                np.random.seed(self.seed)
                # When & Then
                train_gen = LSTM_train.getWindowStart(config)
                with self.assertRaisesRegex(
                        Exception,
                        "No starting positions in the generator."):
                    next(train_gen)

    def test_get_window_start_empty_data(self):
        """Test if fails gracefully in case of no data."""
        # Given
        LSTM_train.training_data = ([], [])
        config = LSTM_train.Config(n_frames=1, frame_step=1)
        np.random.seed(self.seed)
        # When & Then
        train_gen = LSTM_train.getWindowStart(config)
        with self.assertRaisesRegex(
                Exception,
                "No starting positions in the generator."):
            next(train_gen)

    def test_get_window_start_repeat(self):
        """Test if getWindowStart goes through data repeatedly."""
        # Given
        config = LSTM_train.Config(n_frames=2, frame_step=5)
        expected_values = [
            6, 2, 7, 4, 5, 0, 3, 1, 8,
            6, 3, 5, 1, 0, 2, 8, 7, 4
        ]
        np.random.seed(self.seed)
        # When
        train_gen = LSTM_train.getWindowStart(config)
        values = []
        for i in range(len(expected_values)):
            values.append(next(train_gen))
        # Then
        self.assertListEqual(expected_values, values)


class TestCalculateMapping(unittest.TestCase):
    """Test whether the SVD is done correctly."""

    def test_calculate_mapping(self):
        """Test to see if SVD and value sorting works as expected."""
        # Given
        data_folder = pjoin("..", "Database", "TestDatabase")
        data = np.array([
            [1.1, 2.2, 3.3, 4.4, 5.5],
            [1.5, 2.6, 3.7, 4.8, 5.9],
            [2.3, 3.4, 4.5, 5.6, 6.7],
            [3.6, 4.7, 5.8, 6.9, 7.1],
            [4.5, 5.7, 6.9, 7.2, 8.3],
            [5.8, 7.3, 9.2, 11.6, 13.8]
        ], dtype=float)
        expected_S = np.array([33.7448, 2.3277, 0.6587, 0.4985, 0.0207])
        # When
        U, S, V = LSTM_train.calculateMapping(
            data, data_folder, code_test=True)
        # Then
        self.assertTupleEqual((6, 4), U.shape)
        self.assertTupleEqual((4, 5), V.shape)
        # Only k=min(5, 6)-1=4 values are calculated
        np.testing.assert_allclose(S, expected_S[:4], atol=0.001)
        data_recreated = np.dot(U * S, V)  # Dot transposes V
        np.testing.assert_allclose(data, data_recreated, atol=0.05)


class TestLoadMapping(unittest.TestCase):
    """Test to see if mapping is loaded correctly from a file.

    Dependency: calculateMapping (if files don't exist yet)
    """

    def test_load_mapping(self):
        """Test if mapping loads correctly depending on input dimension."""
        # Given
        data_folder = pjoin("..", "Database", "TestDatabase")
        params = [1, 2, 3, 4]  # n_dimension
        for dim in params:
            with self.subTest(dimension=dim):
                config = LSTM_train.Config(n_dimension=dim)
                # When
                mapping_t = LSTM_train.loadMapping(config, data_folder)
                # Then
                self.assertTupleEqual((5, dim), mapping_t.shape)


class TestGetData(unittest.TestCase):

    pass  # TODO: next thing to test


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    LSTM_train.LOGGING = False
    unittest.main()
