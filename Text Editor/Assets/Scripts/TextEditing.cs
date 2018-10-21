using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

public class TextEditing : MonoBehaviour
{
    // TODO: Do geometry calculations to calculate this automatically
    public int max_rows = 15;
    public int max_columns = 53;

    private int chars_in_line = 53;

    private TextMesh full_text;
    private ActivePage page;

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
                return; // Return so that V is not entered as part of the text
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
                model.BuildModel(full_text.text);
            }
        }
    }

    string WrapLines(string text)
    {
        string[] original_lines = text.Split(new char[] {'\n'});

        List<string> wrapped_lines_list = new List<string>();

        foreach (string line in original_lines)
        {
            if (line.Length > chars_in_line)
            {
                string current_line = line;
                while (current_line.Length > chars_in_line)
                {
                    // Find the last space before the character limit
                    int space_index = current_line.LastIndexOf(' ', chars_in_line);
                    if (space_index == -1)
                        space_index = chars_in_line - 1; // -1 to get around the space issue
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
}