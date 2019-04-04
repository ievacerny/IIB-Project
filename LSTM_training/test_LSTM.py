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
                # When
                training, testing = LSTM_train.loadData(
                    testing_prop, self.number_of_vids, self.seed,
                    self.data_folder)
                # Then
                self.assertEqual(2, len(training))
                self.assertEqual(exp_training_len, len(training[0]))
                self.assertListEqual(
                    self.labels[exp_testing_len:], training[1].tolist())

                self.assertEqual(2, len(testing))
                self.assertEqual(exp_testing_len, len(testing[0]))
                self.assertListEqual(
                    self.labels[:exp_testing_len], testing[1].tolist())


class LoadGetWindowStart(unittest.TestCase):
    """Test getWindowStart generator that randomises the input data."""

    def setUp(self):
        """Set up all the tests."""
        self.seed = 3
        number_of_vids = (3+1, 102+1)
        data_folder = pjoin("..", "Database", "TestDatabase")
        testing_prop = 0.2
        training_data, testing_data = LSTM_train.loadData(
            testing_prop, number_of_vids, self.seed, data_folder)
        LSTM_train.training_data = training_data

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
                LSTM_train.n_frames = n_frames
                LSTM_train.frame_step = frame_step
                np.random.seed(self.seed)
                # When
                train_gen = LSTM_train.get_window_start()
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
                LSTM_train.n_frames = n_frames
                LSTM_train.frame_step = frame_step
                np.random.seed(self.seed)
                # When & Then
                train_gen = LSTM_train.get_window_start()
                with self.assertRaisesRegex(
                        Exception,
                        "No starting positions in the generator."):
                    next(train_gen)

    def test_get_window_start_empty_data(self):
        """Test if fails gracefully in case of no data."""
        # Given
        LSTM_train.training_data = ([], [])
        LSTM_train.n_frames = 1
        LSTM_train.frame_step = 1
        np.random.seed(self.seed)
        # When & Then
        train_gen = LSTM_train.get_window_start()
        with self.assertRaisesRegex(
                Exception,
                "No starting positions in the generator."):
            next(train_gen)

    def test_get_window_start_repeat(self):
        """Test if get_window_start goes through data repeatedly."""
        # Given
        LSTM_train.n_frames = 2
        LSTM_train.frame_step = 5
        expected_values = [
            6, 2, 7, 4, 5, 0, 3, 1, 8,
            6, 3, 5, 1, 0, 2, 8, 7, 4
        ]
        np.random.seed(self.seed)
        # When
        train_gen = LSTM_train.get_window_start()
        values = []
        for i in range(len(expected_values)):
            values.append(next(train_gen))
        # Then
        self.assertListEqual(expected_values, values)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    LSTM_train.LOGGING = False
    unittest.main()
