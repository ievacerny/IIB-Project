using System;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;

class TCPClient
{
    string serverIpAddressString = "localhost";
    int port = 65432;
    //string message = "Hey there from the client. Please respond.";
    TcpClient client;

    public TCPClient()
    {
        client = Connect(serverIpAddressString, port);

        if (!client.Connected)
        {
            Debug.Log("Failed to connect.");
            return;
        }
    }

    ~TCPClient()
    {
        Debug.Log("Closing the socket");
        try
        {
            client.Close();
        }
        catch (Exception e)
        {
            Debug.Log(e.Message);
        }
    }

    private TcpClient Connect(String server, int port)
    {
        TcpClient client = new TcpClient();

        try
        {
            client = new TcpClient(server, port);
        }
        catch (ArgumentNullException e)
        {
            Console.WriteLine("ArgumentNullException: {0}", e);
        }
        catch (SocketException e)
        {
            Console.WriteLine("SocketException: {0}", e);
        }

        return client;
    }

    public void SendMessage(string message)
    {
        
        // Translate the passed message into ASCII and store it as a Byte array.
        Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);
        
        // Get a client stream for writing.
        NetworkStream stream = client.GetStream();
        
        // Send the message to the connected TcpServer. 
        stream.Write(data, 0, data.Length);
        
        // Report what was sent to console
        //Debug.Log("Sent: " + message);

        // Flush the stream
        stream.Flush();
        
    }

    public void ReadMessage()
    {
        
        // Buffer to store the response bytes.
        Byte[] data = new Byte[1024];
        
        // String to store the response ASCII representation.
        String responseData = String.Empty;
        
        // Get a client stream for writing.
        NetworkStream stream = client.GetStream();
        
        // Read the first batch of the TcpServer response bytes.
        Int32 bytes = stream.Read(data, 0, data.Length);
        responseData = System.Text.Encoding.ASCII.GetString(data, 0, bytes);
        

        // Report what was received to console
        Debug.Log("Received: " + responseData);
    }
}
