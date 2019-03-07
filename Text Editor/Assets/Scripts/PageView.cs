using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class PageView : MonoBehaviour {

    #region Presenter Reference
    private PagePresenter presenter;
    #endregion
    #region Serialized Attributes
    [SerializeField] private GameObject feedback;
    [SerializeField] private GameObject selection_prefab;
    [SerializeField] private Transform mouse_cursor;
    [SerializeField] private Transform text_cursor;
    [SerializeField] private TextMesh rendered_text;
    [SerializeField] public Renderer frame;
    [SerializeField] private float feedback_duration = 1f;
    #endregion
    #region Public Attributes
    public bool selection_mode = false;
    public bool trigger_active = false;
    public int gesture_code = 0;
    #endregion
    #region Private Parameters
    private readonly float col_margin = 0.01824f;
    private readonly float row_margin = 0.0105f;
    private readonly float width = 0.01806f;
    private readonly float height = 0.0626f;
    private readonly float object_z = -0.00001f;
    private readonly float selection_time_delay = 0.3f;
    private readonly float default_collider_scale = 0.02f;
    private readonly string[] gesture_map = new string[] { "", "Undo", "Redo", "Copy", "Paste", "Delete" };
    public float extended_collider_scale = 0.1f;
    #endregion
    #region Private Attributes
    private List<GameObject> selection_drawing;
    private Indices selection_start;
    private float selection_timer = 0f;
    private float feedback_timer = 0f;
    private bool past_selection_mode = false;
    private Transform trigger_finger;
    private UnityEngine.UI.Text text;
    #endregion

    #region Unity Control

    void Start()
    {
        presenter = new PagePresenter(this, rendered_text.text);

        if (text_cursor == null)
            foreach (Transform child in transform)
            {
                if (child.CompareTag("Text Cursor"))
                    text_cursor = child;
            }

        Assert.IsNotNull(text_cursor, "No text cursor associated with the page");

        if (feedback != null)
            text = feedback.GetComponentInChildren<UnityEngine.UI.Text>();
        else
            Debug.LogError("Feedback system reference is not provided.");

        this.enabled = false;
    }

    private void OnEnable()
    {
        ShowTextCursor(true);
    }

    private void OnDisable()
    {
        presenter.Reset();
        ShowTextCursor(false);
        selection_start = null;
        selection_timer = 0f;
        trigger_active = false;
        text.text = "";
        feedback.SetActive(false);
        if (selection_mode)
        {
            RemoveSelection();
            selection_mode = false;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.name == "IndexTip")
        {
            trigger_active = true;
            trigger_finger = other.transform;
            Vector3 finger_coords = transform.InverseTransformPoint(trigger_finger.position);
            presenter.Click(CoordsToInd(finger_coords));
            SetFrameTransparency(1f);
            ShowTextCursor(true);
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.gameObject.name == "IndexTip" && trigger_active && selection_mode)
        {
            Vector3 finger_coords = transform.InverseTransformPoint(other.transform.position);
            presenter.ClickDrag(selection_start, CoordsToInd(finger_coords), true);
        }
        else if (other.gameObject.name == "IndexTip" && !trigger_active && !selection_mode)
        {
            SetFrameTransparency(0.5f);
            trigger_active = false;
            trigger_finger = null;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.name == "IndexTip")
        {
            if (selection_mode)
                ChangeSelectionModeStatus(false);

            SetFrameTransparency(0.5f);
            trigger_active = false;
            trigger_finger = null;
        }
    }

    void Update()
    {
        #region Feedback and gesture control
        if (gesture_code == 0)
        {
            if (feedback_timer == 0f)
                return;
            if (Time.time - feedback_timer > feedback_duration)
            {
                feedback.SetActive(false);
                text.text = "";
                feedback_timer = 0f;
            }
        }
        else
        {
            if (!feedback.activeSelf)
                feedback.SetActive(true);
            if (text.text != gesture_map[gesture_code])
            {
                text.text = gesture_map[gesture_code];
                Debug.Log(string.Format("Do action {0}", gesture_code));
                switch (gesture_code)
                {
                    case 1:
                        presenter.Undo();
                        break;
                    case 2:
                        presenter.Redo();
                        break;
                    case 3:
                        presenter.Copy();
                        break;
                    case 4:
                        presenter.Paste();
                        break;
                    case 5:
                        presenter.InputString("\b");
                        break;
                }
            }
            feedback_timer = Time.time;
        }
        #endregion

        #region Mouse and keyboard controls
        if (mouse_cursor != null)
        {
            Vector3 mouse_coords = transform.InverseTransformPoint(mouse_cursor.position);
            if (IsMouseCursorOnPage(mouse_coords))
            {
                if (Input.GetMouseButtonDown(0))
                {
                    selection_start = CoordsToInd(mouse_coords);
                    selection_timer = Time.time;
                    ShowTextCursor(false);
                }

                if (Input.GetMouseButton(0) && selection_timer != 0f &&
                    Time.time - selection_timer > selection_time_delay)
                {
                    presenter.ClickDrag(selection_start, CoordsToInd(mouse_coords), true);
                }

                if (Input.GetMouseButtonUp(0) && selection_start != null)
                {
                    if (Time.time - selection_timer > selection_time_delay)
                    {
                        presenter.ClickDrag(selection_start, CoordsToInd(mouse_coords), false);
                    }
                    else
                    {
                        presenter.Click(CoordsToInd(mouse_coords));
                        ShowTextCursor(true);
                    }
                    selection_start = null;
                    selection_timer = 0f;
                }
            }
        }

        // Arrow keys
        if (Input.GetKeyDown(KeyCode.LeftArrow))
            presenter.ArrowKeys(KeyCode.LeftArrow);
        else if (Input.GetKeyDown(KeyCode.RightArrow))
            presenter.ArrowKeys(KeyCode.RightArrow);
        else if (Input.GetKeyDown(KeyCode.UpArrow))
            presenter.ArrowKeys(KeyCode.UpArrow);
        else if (Input.GetKeyDown(KeyCode.DownArrow))
            presenter.ArrowKeys(KeyCode.DownArrow);

        // Control(Shift) events
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            if (Input.GetKeyDown(KeyCode.C))
            {
                presenter.Copy();
                return;  // Return so that the letter is not entered as part of input string
            }
            else if (Input.GetKeyDown(KeyCode.V))
            {
                presenter.Paste();
                return;
            }
            else if (Input.GetKeyDown(KeyCode.Z))
            {
                presenter.Undo();
                return;
            }
            else if (Input.GetKeyDown(KeyCode.Y))
            {
                presenter.Redo();
                return;
            }
        }

        // Input string
        if (Input.inputString.Length != 0)
            presenter.InputString(Input.inputString);
        #endregion
    }

    #endregion

    #region Read Methods

    public string GetRenderedText()
    {
        return rendered_text.text;
    }

    #endregion

    #region Update Methods

    public void UpdateRenderedText(string text)
    {
        rendered_text.text = text;
    }
        
    public void SetTextCursorPosition(Indices ind)
    {
        text_cursor.localPosition = IndToCoords(
            ind.row, ind.col + 0.5f, object_z);
    }

    public void ShowTextCursor(bool show)
    {
        text_cursor.gameObject.SetActive(show);
    }

    public void RemoveSelection()
    {
        if (selection_drawing != null)
        {
            foreach (GameObject drawing in selection_drawing)
                Destroy(drawing);

            selection_drawing = null;
        }
    }

    public void DrawSelection(Indices start, Indices end, int max_line_length)
    {
        RemoveSelection();
        selection_drawing = new List<GameObject>();

        // Flip the Indices if they are the wrong way round
        if (end.IsBefore(start))
        {
            Indices temp_ind = start;
            start = end;
            end = temp_ind;
        }
        
        int no_lines = end.row - start.row + 1;

        float center_index; int selection_length; Vector3 pos;

        // In-line selection (for single line)
        if (no_lines == 1)
        {
            center_index = (float)(end.col - start.col) / 2.0f + start.col;
            selection_length = end.col - start.col + 1;
            pos = IndToCoords(start.row, center_index, object_z);
            DrawRectangle(pos, selection_length);
            return;
        }

        // Full lines (for multiline)
        if (no_lines > 2)
        {
            for (int i = start.row + 1; i < end.row; i++)
            {
                pos = IndToCoords(i, (float)max_line_length / 2.0f - 0.5f, object_z);
                DrawRectangle(pos, max_line_length);
            }
        }

        // First line (for multiline)
        center_index = (float)(max_line_length - 1 - start.col) / 2.0f + start.col;
        selection_length = max_line_length - start.col;
        pos = IndToCoords(start.row, center_index, object_z);
        DrawRectangle(pos, selection_length);

        // Last line (for multiline)
        center_index = (float)end.col / 2.0f;
        selection_length = end.col + 1;
        pos = IndToCoords(end.row, center_index, object_z);
        DrawRectangle(pos, selection_length);
    }

    public void SetMouseCursor(Transform mouse_cursor)
    {
        if (this.mouse_cursor != null)
            return;

        this.mouse_cursor = mouse_cursor;
    }

    public void SetFrameTransparency(float alpha)
    {
        Color new_color = frame.material.color;
        new_color.a = alpha;
        frame.material.color = new_color;
    }

    public void ChangeSelectionModeStatus(bool status, bool error=false)
    {
        // IMPORTANT: has to be changes before trigger_active
        selection_mode = status;
        Vector3 size = transform.GetComponent<BoxCollider>().size;
        if (status)
        {
            Vector3 finger_coords = transform.InverseTransformPoint(trigger_finger.position);
            selection_start = CoordsToInd(finger_coords);
            size.z = extended_collider_scale;
            transform.GetComponent<BoxCollider>().size = size;
            ShowTextCursor(false);
        }
        else
        {
            if (!error)
            {
                Vector3 finger_coords = transform.InverseTransformPoint(trigger_finger.position);
                presenter.ClickDrag(selection_start, CoordsToInd(finger_coords), false);
            }
            else
            {
                RemoveSelection();
                ShowTextCursor(true);
            }
            size.z = default_collider_scale;
            transform.GetComponent<BoxCollider>().size = size;
        }
    }

    #endregion

    #region Private Methods

    private void DrawRectangle(Vector3 local_position, int x_scaling)
    {
        GameObject line = Instantiate(selection_prefab, transform);
        line.transform.localPosition = local_position;
        Vector3 scale = line.transform.localScale;
        line.transform.localScale = new Vector3(scale.x * x_scaling, scale.y, scale.z);
        selection_drawing.Add(line);
    }

    private bool IsMouseCursorOnPage(Vector3 coords)
    {
        if (coords.x <= 0.5f && coords.x >= -0.5f &&
            coords.y <= 0.5 && coords.y >= -0.5f)
            return true;
        else
            return false;
    }

    #endregion

    #region Coordinate Conversions

    public Indices CoordsToInd(Vector3 coords)
    {
        // x - row, y - col
        Indices ind = new Indices
        {
            //(int)0.8 == (int)-0.8, but floor(0.8) == 0 and floor(-0.8) == -1
            row = Mathf.FloorToInt((coords.y - (0.5f - row_margin)) / -height),
            col = Mathf.FloorToInt((coords.x + (0.5f - col_margin)) / width),
        };
        
        return ind;
    }

    public Vector3 IndToCoords(float row, float col, float z_coord = 0f)
    {
        var coords = new Vector3
        {
            y = (row + 0.5f) * -height + (0.5f - row_margin),
            x = (col + 0.5f) * width - (0.5f - col_margin),
            z = z_coord
        };

        return coords;
    }

    #endregion

    #region Testing Interface

    public void ChangePresenterReference(PagePresenter new_presenter)
    {
        // Needed for testing purposes
        presenter = new_presenter;
    }

    #endregion
}
