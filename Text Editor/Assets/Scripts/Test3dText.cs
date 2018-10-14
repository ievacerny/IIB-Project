using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Test3dText : MonoBehaviour {

    /*  3d-Text Edit
     Editable 3d-text object.
     Put this script on a 3d-text GameObject.
     Note: Make sure to have a collider on the GameObject to register Raycasts
    */

    private bool inEditMode = false;
    private string storedString;
    private TextMesh textComponent;
    private string guiString;
    private Renderer this_renderer;
 
    void Start()
    {
        //Store the String
        textComponent = GetComponent<TextMesh>();
        this_renderer = GetComponent<Renderer>();
        storedString = textComponent.text;
        guiString = storedString;
        Debug.Log(storedString);
        

        //Visual Aid for Focus (example)
        Color cur_color = this_renderer.material.color;
        cur_color.a = 0.5F;
        this_renderer.material.color = cur_color;

        CheckChars();    //Check so that the 3d-text isn't empty
        FitCollider();    //Set the Collider to fit the 3d Text Size
    }

    //void OnGUI()
    //{
    //    if (inEditMode)
    //    {
    //        //Make a TextField which sends to the 3d-text GameObject
    //        GUI.SetNextControlName("hiddenTextField"); //Prepare a Control Name so we can focus the TextField
    //        GUI.FocusControl("hiddenTextField");        //Focus the TextField
    //        guiString = GUI.TextField(new Rect(90, -100, 200, 25), guiString, 25);    //Display a TextField outside the Screen Rect
    //        //Listen for keys
    //        if (Input.anyKey)
    //        {
    //            Debug.Log("key detected");
    //            textComponent.text = guiString;    //Set the 3d-text to the same as our hidden TextField
    //            Debug.Log(textComponent.text);
    //            FitCollider();     //Resize the Collider
    //            Debug.Log("Done listening");
    //            Debug.Log(guiString);
    //        }
    //        Debug.Log(Input.inputString);
    //    }
    //}

    void Update()
    {
        if (inEditMode)
        {
            // TODO: Add detection of CTRL+V at some point
            if (Input.inputString.Length != 0)
            {
                //if (Input.inputString.Contains('\n') || Input.inputString.Contains('\r'))
                //{

                //}
                string cur_string = textComponent.text + Input.inputString;
                textComponent.text = cur_string;
            }
        }

        //Begin Edit on RightClick
        if (Input.GetMouseButtonDown(1))
        {
            Debug.Log("click");
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                Debug.Log("hit");
                Debug.Log(hit.transform);
                Debug.Log(transform);
                if (hit.transform == transform)
                {
                    Debug.Log("Hit transform");
                    inEditMode = true;
                    Color cur_color = this_renderer.material.color;
                    cur_color.a = 1F;
                    this_renderer.material.color = cur_color;
                }
                else
                {
                    inEditMode = false;
                    Color cur_color = this_renderer.material.color;
                    cur_color.a = 0.5F;
                    this_renderer.material.color = cur_color;
                    CheckChars();    //Check so the 3d-text isn't empty
                }
            }
        }
        //Exit Edit on KeyCode Return or Escape
        if (inEditMode && Input.GetKeyDown(KeyCode.Escape))
        {
            inEditMode = false;
            Color cur_color = this_renderer.material.color;
            cur_color.a = 0.5F;
            this_renderer.material.color = cur_color;
            CheckChars();    //Check so the 3d-text isn't empty
        }
    }

    //Set the Collider to fit the 3d Text Size
    void FitCollider()
    {
        BoxCollider collider = GetComponent<BoxCollider>();
        Vector3 cur_size = collider.size;
        cur_size.x = this_renderer.bounds.size.x;
        cur_size.y = this_renderer.bounds.size.y;
        collider.size = cur_size;
    }

    //Check the Size of the 3d-text
    void CheckChars()
    {
        if (textComponent.text.ToCharArray().Length == 0)
        {
            textComponent.text = "NULL";
            FitCollider();
        }
    }

    /*
    //If you want inEditMode on Left Click use this function instead of manual Raycasting
    function OnMouseDown () {
        inEditMode = true;
        }
    */
}
