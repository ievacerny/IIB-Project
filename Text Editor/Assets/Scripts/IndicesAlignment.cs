using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class IndicesAlignment : MonoBehaviour {

    /*
    Script is to be used when the character indices get misaligned.
    Attach the script to the COPY of the page prefab. The script will create bunch of
    dot objects attached to the page. Delete all those objects every time the script is changed
    (recompiled)
    */

    [SerializeField] GameObject dot;
    [Range(0.015f, 0.02f)] public float width = 0.0175f;
    [Range(0.05f, 0.07f)] public float height = 0.06f;
    [Range(0.015f, 0.025f)] public float col_margin = 0.02f;
    [Range(0f, 0.015f)] public float row_margin = 0.01f;
    private GameObject[,] dots = new GameObject[15, 54];
    private bool initialised = false;

    private void OnValidate()
    {
        if (!initialised)
            Initialise();

        for (int i = 0; i < 15; i++)
        {
            for (int j = -1; j < 53; j++)
            {
                try
                {
                    dots[i, j + 1].transform.localPosition = IndToCoords(i, j);
                }
                catch (System.NullReferenceException)
                {
                    initialised = false;
                    return;
                }
                catch (MissingReferenceException)
                {
                    initialised = false;
                    return;
                }
            }
        }
    }

    private Vector3 IndToCoords(float row, float col, float z_coord = 0f)
    {
        var coords = new Vector3
        {
            y = (row + 0.5f) * -height + (0.5f - row_margin),
            x = (col + 0.5f) * width - (0.5f - col_margin),
            z = z_coord
        };

        return coords;
    }

    private void Initialise()
    {
        initialised = true;
        for (int i = 0; i < 15; i++)
        {
            for (int j = -1; j < 53; j++)
            {
                GameObject new_dot = Instantiate(dot, transform);
                new_dot.transform.localPosition = IndToCoords(i, j);
                dots[i, j+1] = new_dot;
            }
        }
    }

}
