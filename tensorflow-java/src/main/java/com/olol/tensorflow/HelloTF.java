package com.olol.tensorflow;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class HelloTF {
    public static void main(String[] args) throws Exception {
        try (Graph g = new Graph()) {
            final String value = "Hello from " + TensorFlow.version();

            // Construct the computation graph with a single operation, a constant
            // named "MyConst" with a value "value".
            try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
                // The Java API doesn't yet include convenience functions for adding operations.
                g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
            }

            // Execute the "MyConst" operation in a Session.
            try (Session s = new Session(g);
                 Tensor output = s.runner().fetch("MyConst").run().get(0)) {
                System.out.println(new String(output.bytesValue(), "UTF-8"));
            }
        }

        try (Graph g = new Graph()) {
            String modelDir = HelloTF.class.getClassLoader().getResource("SNN_294_100_sigmoid_softmax.pb").getPath();
            final byte[] graphDef = Files.readAllBytes(Paths.get(modelDir));
            g.importGraphDef(graphDef);
            System.out.println(g.operation("dense_5_input_2").output(0));
            System.out.println(g.operation("output_node0").output(0));
            System.out.println(g.operation("strided_slice_1").output(0));
        }
    }
}
