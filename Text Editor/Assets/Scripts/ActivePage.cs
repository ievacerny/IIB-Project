using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class ActivePage : MonoBehaviour {

    public bool inEditMode;

    public GameObject selection_prefab;
    public GameObject frame_quad;
    private Renderer frame_quad_rend;
    private Transform cursor_transform;

    private bool cursor_on_page = false;
    public float selection_time_delay = 0.3f;
    private float selection_timer = 0f;

    private float margin = 0.00575f;
    private float width = 0.0186f;
    private float height = 0.0643f;

    private List<GameObject> selection_drawing;

    private Indices start;

    public Model model;
    ModelViewMapping modelview;

    void Start ()
    {
        if (!frame_quad)
        {
            // There should be only one frame in a page. If not, it will grab the first one
            foreach (Transform child in transform) if (child.CompareTag("Frame"))
                    frame_quad = child.gameObject;
        }

        Assert.IsNotNull(frame_quad);
        frame_quad_rend = frame_quad.GetComponent<MeshRenderer>();
        Color new_color = frame_quad_rend.material.color;
        new_color.a = 0f;
        frame_quad_rend.material.color = new_color;

        modelview = new ModelViewMapping(margin, width, height);
    }
	
	void Update ()
    {
        // Selection checks
        if (cursor_transform != null && inEditMode)
        {
            Vector3 coords = transform.InverseTransformPoint(cursor_transform.position);
            if (coords.x <= 0.5f && coords.x >= -0.5f && coords.y <= 0.5 && coords.y >= -0.5f)
            {
                if (Input.GetMouseButtonDown(0))
                {
                    start = modelview.CoordsToIndex(coords);
                    selection_timer = Time.time;
                }

                if (Input.GetMouseButton(0) && selection_timer != 0f && 
                    Time.time - selection_timer > selection_time_delay)
                {
                    DrawSelection(start, modelview.CoordsToIndex(coords));
                }

                if (Input.GetMouseButtonUp(0) && start != null)
                {
                    if (Time.time - selection_timer > selection_time_delay)
                    {
                        var end = modelview.CoordsToIndex(coords);
                        Debug.Log(model.GetSelection(start, end));
                        DrawSelection(start, end);
                    }
                    else
                    {
                        RemoveSelection();
                    }
                    start = null;
                    selection_timer = 0f;
                }

                //var ind = modelview.CoordsToIndex(coords);
                //var coords2 = modelview.IndexToCoords(ind);
                //Debug.Log(string.Format("Real {0}, new {1}; row {2}, col {3}", coords.ToString("F4"), coords2.ToString("F4"), ind.row, ind.col));
                //Debug.Log(model.GetLetter(ind));
                //Debug.Log(model.GetWord(ind));
            }
        }


        if (!inEditMode && cursor_on_page && Input.GetMouseButtonDown(0))
        {
            inEditMode = true;
            Color new_color = frame_quad_rend.material.color;
            new_color.a = 1f;
            frame_quad_rend.material.color = new_color;
        }
        else if (inEditMode && cursor_on_page && Input.GetKeyDown(KeyCode.Escape))
        {
            inEditMode = false;
            Color new_color = frame_quad_rend.material.color;
            new_color.a = 0.5f;
            frame_quad_rend.material.color = new_color;
        }
        else if (inEditMode && !cursor_on_page && Input.GetKeyDown(KeyCode.Escape))
        {
            inEditMode = false;
            Color new_color = frame_quad_rend.material.color;
            new_color.a = 0f;
            frame_quad_rend.material.color = new_color;
        }
    }


    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.name == "FingerPoint")
        {
            cursor_on_page = true;
            cursor_transform = other.transform;

            if (!inEditMode)
            {
                Color new_color = frame_quad_rend.material.color;
                new_color.a = 0.5f;
                frame_quad_rend.material.color = new_color;
            }
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.name == "FingerPoint")
        {
            cursor_on_page = false;
            cursor_transform = null;

            if (!inEditMode)
            {
                Color new_color = frame_quad_rend.material.color;
                new_color.a = 0f;
                frame_quad_rend.material.color = new_color;
            }
        }
    }


    private void DrawSelection(Indices start, Indices end)
    {
        if (selection_drawing != null)
        {
            foreach (GameObject drawing in selection_drawing)
                Destroy(drawing);
        }
        selection_drawing = new List<GameObject>();
        
        Vector3 pos;
        float center_index;

        int no_lines = Mathf.Abs(start.row - end.row) + 1;

        if (end.IsBefore(start))
        {
            Indices temp_ind = start;
            start = end;
            end = temp_ind;
        }

        // In-line selection
        if (no_lines == 1)
        {
            center_index = (float)(end.col - start.col) / 2.0f + start.col;
            pos = modelview.IndexToCoords(start.row, center_index);
            DrawRectangle(new Vector3(pos.x, pos.y, -0.01f), (end.col - start.col + 1));
            return;
        }

        // Full lines (for multiline)
        if (no_lines > 2)
        {
            for (int i = start.row + 1; i < end.row; i++)
            {
                pos = modelview.IndexToCoords(i, (float)model.max_columns / 2.0f - 0.5f);
                DrawRectangle(new Vector3(pos.x, pos.y, -0.01f), model.max_columns);
            }
        }

        // First line (for multiline)
        center_index = (float)(model.max_columns - 1 - start.col) / 2.0f + start.col;
        pos = modelview.IndexToCoords(start.row, center_index);
        DrawRectangle(new Vector3(pos.x, pos.y, -0.01f), (model.max_columns - start.col));
        
        // Last line (for multiline)
        center_index = (float) end.col / 2.0f;
        pos = modelview.IndexToCoords(end.row, center_index);
        DrawRectangle(new Vector3(pos.x, pos.y, -0.01f), (end.col + 1));
    }

    private void DrawRectangle(Vector3 local_position, int x_scaling)
    {
        GameObject line = Instantiate(selection_prefab, transform);
        line.transform.localPosition = local_position;
        Vector3 scale = line.transform.localScale;
        line.transform.localScale = new Vector3(scale.x * x_scaling, scale.y, scale.z);
        selection_drawing.Add(line);
    }

    private void RemoveSelection()
    {
        if (selection_drawing != null)
        {
            foreach (GameObject drawing in selection_drawing)
                Destroy(drawing);
        }
        selection_drawing = null;
    }

}
