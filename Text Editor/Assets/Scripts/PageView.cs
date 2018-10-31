using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class PageView : MonoBehaviour {

    #region Presenter Reference
    private PagePresenter presenter;
    #endregion
    #region Serialized Attributes
    [SerializeField] private GameObject selection_prefab;
    [SerializeField] private Transform mouse_cursor;
    [SerializeField] private Transform text_cursor;
    [SerializeField] private TextMesh rendered_text;
    #endregion
    #region Private Parameters
    private readonly float col_margin = 0.00575f;
    private readonly float width = 0.0186f;
    private readonly float height = 0.0643f;
    private readonly float object_z = -0.1f;
    private readonly float selection_time_delay = 0.3f;
    #endregion
    #region Private Attributes
    private List<GameObject> selection_drawing;
    private Indices selection_start;
    private float selection_timer = 0f;
    #endregion

    #region Unity Control

    void Start () {
        presenter = new PagePresenter(this, rendered_text.text);

        if (text_cursor == null)
            foreach (Transform child in transform)
            {
                if (child.CompareTag("Text Cursor"))
                    text_cursor = child;
            }

        Assert.IsNotNull(text_cursor, "No text cursor associated with the page");
        text_cursor.gameObject.SetActive(false);
    }
	
	void Update () {
        
        // Mouse clicks
        if (mouse_cursor != null)
        {
            Vector3 mouse_coords = transform.InverseTransformPoint(mouse_cursor.position);
            if (IsMouseCursorOnPage(mouse_coords))
            {
                if (Input.GetMouseButtonDown(0))
                {
                    selection_start = CoordsToInd(mouse_coords);
                    selection_timer = Time.time;
                    text_cursor.gameObject.SetActive(false);
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
                        text_cursor.gameObject.SetActive(true);
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
                presenter.Copy();
            else if (Input.GetKeyDown(KeyCode.V))
                presenter.Paste();
            else if (Input.GetKeyDown(KeyCode.Z))
                presenter.Undo();
            else if (Input.GetKeyDown(KeyCode.Y))
                presenter.Redo();

            // Return so that the letter is not entered as part of input string
            return;
        }

        // Input string
        if (Input.inputString.Length != 0)
        {
            presenter.InputString(Input.inputString);
        }
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
        
        int no_lines = start.row - end.row + 1;

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
            row = (int)((coords.y - 0.5f) / -height),
            col = (int)((coords.x + (0.5f - col_margin)) / width),
        };
        
        return ind;
    }

    public Vector3 IndToCoords(float row, float col, float z_coord = 0f)
    {
        var coords = new Vector3
        {
            y = (row + 0.5f) * -height + 0.5f,
            x = (col + 0.5f) * width - (0.5f - col_margin),
            z = z_coord
        };

        return coords;
    }

    #endregion
}
