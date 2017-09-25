package com.olol.tensorflow;

import org.tensorflow.*;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

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

        try (Graph g = new Graph(); Session s = new Session(g)) {
            String modelDir = HelloTF.class.getClassLoader().getResource("SNN_294_100_sigmoid_softmax.pb").getPath();
            final byte[] graphDef = Files.readAllBytes(Paths.get(modelDir));
            g.importGraphDef(graphDef);
            System.out.println(g.operation("dense_5_input_2").output(0));
            System.out.println(g.operation("output_node0").output(0));
            System.out.println(g.operation("strided_slice_1").output(0));




            long[] shape = {294};

            List<String> lines = Files.readAllLines(Paths.get(HelloTF.class.getClassLoader().getResource("data/test_data_s01_s_jan_dec_2015_298F_edited_head").getPath()));
            for (String line: lines.subList(1, 10)) {
                String[] values = line.split("\\|");
                Float[][] floats = new Float[1][294];
                for (int i = 0; i < values.length - 2; i++) {
                    floats[0][i] = Float.parseFloat(values[i]);
//                    floats[0][i] = 0.2f;
                }
                Tensor inputs = Tensor.create(floats);
                float[][] vals = inputs.copyTo(new float[1][294]);
                System.out.println(vals[0][]);

                Tensor output = s.runner().feed("dense_5_input_2", inputs).fetch("output_node0").run().get(0);
                float[] val = output.copyTo(new float[2]);
                System.out.println(val[0] + " " + val[1]);


//                Tensor inputs = new Tensor("dense_5_input_2", floats)
//                session.runner().feed("dense_5_input_2", floats)
            }

        }
    }
}
