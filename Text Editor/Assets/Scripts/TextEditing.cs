using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;

public class TextEditing : MonoBehaviour
{
    // TODO: Do geometry calculations to calculate this automatically
    public int max_rows = 15;
    public int max_columns = 53;

    private TextMesh full_text;
    private ActivePage page;

    private Indices selected_text_ind;
    private string selected_text =  "";

    Model model;

    void Start()
    {
        full_text = GetComponent<TextMesh>();
        page = transform.GetComponentInParent<ActivePage>();
        model = new Model(full_text.text, max_rows, max_columns);
        page.model = model;
    }

    void Update()
    {
        if (page.inEditMode)
        {
            if ((Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) && Input.GetKeyDown(KeyCode.V))
            {
                string new_text = GUIUtility.systemCopyBuffer;
                full_text.text = WrapLines(full_text.text + new_text);
                model.BuildModel(full_text.text, max_rows, max_columns);
                // Deselect previously selected text
                DeselectText();
                page.RemoveSelection();
                return; // Return so that V is not entered as part of the text
            }

            if ((Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) && Input.GetKeyDown(KeyCode.C))
            {
                if (selected_text == "")
                    Debug.Log("No text selected");
                else
                    GUIUtility.systemCopyBuffer = selected_text;
                return; // Return so that C is not entered as part of the text
            }

            if (Input.inputString.Length != 0)
            {
                string new_text = full_text.text;

                foreach (char c in Input.inputString)
                {
                    // \b, \n and \r are the only supported characters in inputString
                    if (c == '\b') // has backspace/delete been pressed?
                    {
                        if (new_text.Length != 0)
                            new_text = new_text.Substring(0, new_text.Length - 1);
                        if (selected_text != "")
                        {
                            new_text = DeleteSelectedText();
                            DeselectText();
                            page.RemoveSelection();
                            break;
                        }
                    }
                    else if ((c == '\n') || (c == '\r')) // enter/return
                    {
                        new_text = new_text + '\n';
                    }
                    else
                    {
                        new_text += c;
                    }
                }

                full_text.text = WrapLines(new_text);
                model.BuildModel(full_text.text, max_rows, max_columns);
            }
        }
    }

    private string WrapLines(string text)
    {
        string[] original_lines = text.Split(new char[] {'\n'});

        List<string> wrapped_lines_list = new List<string>();

        foreach (string line in original_lines)
        {
            if (line.Length > max_columns)
            {
                string current_line = line;
                while (current_line.Length > max_columns)
                {
                    // Find the last space before the character limit
                    int space_index = current_line.LastIndexOf(' ', max_columns);
                    if (space_index == -1)
                        space_index = max_columns - 1; // -1 to get around the space issue
                    wrapped_lines_list.Add(current_line.Substring(0, space_index+1));
                    current_line = current_line.Substring(space_index+1);
                }
                wrapped_lines_list.Add(current_line);
            }
            else
            {
                wrapped_lines_list.Add(line);
            }
        }

        string wrapped_lines = string.Join("\n", wrapped_lines_list.ToArray());

        return wrapped_lines;
    }

    public void SelectText(Indices start, Indices end)
    {
        selected_text = model.GetSelection(start, end);
        selected_text_ind = start;
        Debug.Log(selected_text);
    }

    public void DeselectText()
    {
        selected_text = "";
        selected_text_ind = null;
        Debug.Log("Deselected");
    }

    private string DeleteSelectedText()
    {
        int index = model.IndicesToStringIndex(selected_text_ind);
        return full_text.text.Remove(index, selected_text.Length);
    }
}